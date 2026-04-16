import io
import os
import sys
import time
import uuid
import logging
from typing import List, Optional
from urllib.parse import unquote

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import fitz  # PyMuPDF

# Optional OCR support for scanned PDFs
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")  # Suppress HuggingFace noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")
TESSERACT_CMD    = os.environ.get("TESSERACT_CMD")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing. Please add it to your .env file.")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is missing. Please add it to your .env file.")

if OCR_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INDEX_NAME        = "pdf-rag-index"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
TOP_K             = 8
MIN_CHUNK_LEN     = 40
MAX_CONTEXT_CHARS = 14000
GROQ_MODEL        = "llama-3.3-70b-versatile"
UPSERT_BATCH_SIZE = 100
INDEX_READY_POLL  = 1
POST_UPSERT_WAIT  = 5

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"❌ CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})."
    )


# ─────────────────────────────────────────────
# CONFIG — image OCR filtering
# ─────────────────────────────────────────────
MIN_IMAGE_PIXELS = 100 * 100   # skip tiny icons/bullets (width×height)
MIN_IMAGE_OCR_LEN = 20         # discard OCR results shorter than this


def _ocr_image_bytes(img_bytes: bytes, label: str) -> str:
    """
    Run pytesseract on raw image bytes.
    Returns stripped OCR text, or "" on failure.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img).strip()
        if len(text) >= MIN_IMAGE_OCR_LEN:
            return text
    except Exception as e:
        log.warning(f"   ⚠️  OCR failed on {label}: {e}")
    return ""


def _extract_embedded_image_texts(page: fitz.Page, page_num: int) -> List[str]:
    """
    Extract every image embedded in a PDF page and OCR it.
    Skips images that are too small (icons, decorations).
    Returns a list of non-empty OCR strings.
    """
    if not OCR_AVAILABLE:
        return []

    texts = []
    image_list = page.get_images(full=True)  # [(xref, smask, w, h, ...), ...]

    for img_index, img_info in enumerate(image_list):
        xref   = img_info[0]
        width  = img_info[2]
        height = img_info[3]

        if width * height < MIN_IMAGE_PIXELS:
            log.debug(f"   Skipping tiny image {img_index+1} on page {page_num} ({width}×{height}px)")
            continue

        try:
            base_image = page.parent.extract_image(xref)
            img_bytes  = base_image["image"]
            label      = f"image {img_index+1} on page {page_num}"
            ocr_text   = _ocr_image_bytes(img_bytes, label)

            if ocr_text:
                log.info(f"   🖼️  OCR extracted {len(ocr_text)} chars from {label}")
                texts.append(ocr_text)
        except Exception as e:
            log.warning(f"   ⚠️  Could not extract image {img_index+1} on page {page_num}: {e}")

    return texts


# ─────────────────────────────────────────────
# 1. PDF LOADING & CHUNKING
# ─────────────────────────────────────────────
def load_pdf(pdf_path: str) -> List[dict]:
    pdf_path = unquote(pdf_path)
    pdf_path = os.path.normpath(pdf_path)

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: List[dict] = []
    scanned_pages = 0
    total_embedded_images = 0

    for i, page in enumerate(doc):
        page_num  = i + 1
        page_text = page.get_text().strip()

        # ── Embedded image OCR (runs on ALL pages, not just image-only ones) ──
        image_texts = _extract_embedded_image_texts(page, page_num)
        total_embedded_images += len(image_texts)

        if page_text:
            # Combine selectable text + any embedded image OCR on the same page
            combined = page_text
            if image_texts:
                combined += "\n\n[Image content on this page]:\n" + "\n\n".join(image_texts)
            pages.append({"text": combined, "page": page_num})

        else:
            # Fully scanned/image-only page — OCR the whole page render
            scanned_pages += 1
            if OCR_AVAILABLE:
                try:
                    pix      = page.get_pixmap(dpi=200)
                    ocr_text = _ocr_image_bytes(pix.tobytes("png"), f"full page {page_num}")

                    # Also merge any individually extracted embedded images
                    all_text_parts = ([ocr_text] if ocr_text else []) + image_texts
                    combined = "\n\n".join(all_text_parts).strip()

                    if combined:
                        log.info(f"   🔍 OCR applied on full page {page_num}")
                        pages.append({"text": combined, "page": page_num})
                    else:
                        log.warning(f"   ⚠️  Page {page_num}: OCR found no text (blank or unreadable).")
                except Exception as e:
                    log.error(f"   ❌ OCR failed on page {page_num}: {e}")
            else:
                log.warning(f"   ⚠️  Page {page_num} has no selectable text and OCR is unavailable.")

    doc.close()

    if scanned_pages > 0 and not OCR_AVAILABLE:
        log.warning(
            f"\n⚠️  {scanned_pages} page(s) were image-based and skipped.\n"
            "   To enable OCR: pip install pytesseract pillow and set TESSERACT_CMD in .env\n"
        )

    if total_embedded_images > 0:
        log.info(f"   🖼️  OCR processed {total_embedded_images} embedded image(s) across all pages.")

    log.info(f"   → Extracted text from {len(pages)} page(s) ({scanned_pages} image-only skipped)")
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size}).")

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - overlap

    return chunks


def chunk_pages(pages: List[dict], **kwargs) -> List[dict]:
    records: List[dict] = []
    for page in pages:
        page_chunks = chunk_text(page["text"], **kwargs)
        for i, chunk in enumerate(page_chunks):
            records.append({"text": chunk, "page": page["page"], "chunk_index": i})
    return records


# ─────────────────────────────────────────────
# 2. EMBEDDINGS
# ─────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        log.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True, batch_size=64).tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]


# ─────────────────────────────────────────────
# 3. PINECONE VECTOR STORE
# ─────────────────────────────────────────────
class PineconeStore:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(INDEX_NAME)

    def _ensure_index(self):
        existing_names = [idx.name for idx in self.pc.list_indexes()]
        if INDEX_NAME not in existing_names:
            log.info(f"Creating Pinecone index: {INDEX_NAME}")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            log.info("Waiting for index to be ready...")
            while not self.pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(INDEX_READY_POLL)
            log.info("✅ Index is ready.")
        else:
            log.info(f"Using existing Pinecone index: {INDEX_NAME}")

    def upsert(self, records: List[dict], embeddings: List[List[float]], source: str = "pdf") -> int:
        vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": emb,
                "metadata": {
                    "text": record["text"],
                    "source": source,
                    "page": record["page"],
                    "chunk_index": record["chunk_index"],
                },
            }
            for record, emb in zip(records, embeddings)
        ]

        total_upserted = 0
        total_batches = (len(vectors) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

        for batch_num, i in enumerate(range(0, len(vectors), UPSERT_BATCH_SIZE), start=1):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            try:
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
                log.info(f"   Batch {batch_num}/{total_batches}: upserted {len(batch)} vectors ✓")
            except Exception as e:
                log.error(f"   ❌ Batch {batch_num}/{total_batches} failed: {e}")

        log.info(f"✅ Total upserted: {total_upserted}/{len(vectors)} vectors.")
        return total_upserted

    def query(self, query_embedding: List[float], top_k: int = TOP_K, source_filter: Optional[str] = None) -> List[dict]:
        filter_dict = {"source": {"$eq": source_filter}} if source_filter else None
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )
        return [
            {
                "text": match.metadata["text"],
                "source": match.metadata.get("source", "unknown"),
                "page": match.metadata.get("page", "?"),
                "score": round(match.score, 4),
            }
            for match in results.matches
        ]

    def verify(self) -> int:
        stats = self.index.describe_index_stats()
        if hasattr(stats, "total_vector_count"):
            total = stats.total_vector_count
        elif isinstance(stats, dict):
            total = stats.get("total_vector_count") or stats.get("totalVectorCount", 0)
        else:
            total = 0
        log.info(f"📊 Pinecone index stats: {total} total vectors stored.")
        return total

    def delete_all(self, namespace: str = ""):
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            log.info(f"🗑️  All vectors deleted from namespace '{namespace or 'default'}'.")
        except Exception as e:
            log.error(f"❌ Failed to delete vectors: {e}")


# ─────────────────────────────────────────────
# 4. GROQ LLM
# ─────────────────────────────────────────────
class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def _build_context(self, context_chunks: List[dict]) -> str:
        """Shared helper: format retrieved chunks into a context string."""
        parts = []
        for chunk in context_chunks:
            header = f"[Source: {chunk['source']} | Page {chunk['page']} | Score: {chunk['score']}]"
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_CHARS]

    def generate(self, query: str, context_chunks: List[dict]) -> str:
        """Single-turn: answer from context only."""
        context = self._build_context(context_chunks)

        system_prompt = (
            "You are a precise and helpful assistant. "
            "Answer the user's question using ONLY the provided context below. "
            "If the answer cannot be found in the context, clearly say so — do not guess or hallucinate. "
            "When relevant, mention which page or source the answer comes from."
        )

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"❌ Groq API error: {e}")
            return f"❌ Error generating response: {e}"

    def generate_with_history(
        self,
        query: str,
        context_chunks: List[dict],
        history: Optional[List[dict]] = None,
    ) -> str:
        """
        Multi-turn: includes prior conversation history so the LLM can
        handle follow-up questions that reference previous answers.

        history format: [{"role": "user"|"assistant", "content": "..."}]
        """
        context = self._build_context(context_chunks)

        system_prompt = (
            "You are a precise and helpful assistant. "
            "Answer the user's question using ONLY the provided context below. "
            "You may refer to the conversation history to resolve follow-up questions, "
            "but your answers must still be grounded in the context. "
            "If the answer cannot be found in the context, clearly say so — do not guess or hallucinate. "
            "When relevant, mention which page or source the answer comes from.\n\n"
            f"Context:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Inject prior turns (cap at last 10 to stay within token limits)
        if history:
            messages.extend(history[-10:])

        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"❌ Groq API error: {e}")
            return f"❌ Error generating response: {e}"


# ─────────────────────────────────────────────
# 5. RAG PIPELINE
# ─────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store    = PineconeStore()
        self.llm      = GroqLLM()

    def ingest(self, pdf_path: str, clear_existing: bool = False, namespace: str = ""):
        log.info(f"\n📄 Ingesting: {pdf_path}")

        if clear_existing:
            log.info("🗑️  Clearing existing vectors before ingestion...")
            self.store.delete_all(namespace=namespace)
            time.sleep(2)

        pages = load_pdf(pdf_path)
        if not pages:
            log.error("❌ No text extracted from PDF. Aborting.")
            return

        records = chunk_pages(pages)
        log.info(f"   → {len(records)} chunks from {len(pages)} pages")

        if not records:
            log.error("❌ No valid chunks created. Check PDF content.")
            return

        texts = [r["text"] for r in records]
        log.info(f"   → Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)

        source_name = os.path.basename(pdf_path)
        upserted = self.store.upsert(records, embeddings, source=source_name)

        log.info(f"   ⏳ Waiting {POST_UPSERT_WAIT}s for Pinecone to index...")
        time.sleep(POST_UPSERT_WAIT)

        stored = self.store.verify()
        if stored >= upserted:
            log.info(f"   ✅ Ingestion verified: {stored} vectors in index.\n")
        else:
            log.warning(
                f"   ⚠️  Mismatch: upserted {upserted} but index shows {stored}. "
                "This may resolve shortly (Pinecone eventual consistency).\n"
            )

    def query(self, question: str, source_filter: Optional[str] = None) -> tuple[str, List[dict]]:
        """
        Retrieve relevant chunks and return (answer, chunks).
        Returning chunks allows app.py to display source citations.

        FIX: Previously returned only a string — now returns a tuple so
        app.py doesn't have to re-implement retrieval itself.
        """
        log.info(f"\n🔍 Query: {question}")
        q_embedding = self.embedder.embed_one(question)
        chunks = self.store.query(q_embedding, source_filter=source_filter)

        if not chunks:
            return (
                "⚠️ No relevant content found. Make sure you have ingested a PDF first, "
                "or try rephrasing your question.",
                [],
            )

        log.info(f"   → Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']})")
        answer = self.llm.generate(question, chunks)
        return answer, chunks

    def query_with_history(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        source_filter: Optional[str] = None,
    ) -> tuple[str, List[dict]]:
        """
        Multi-turn variant used by app.py to support follow-up questions.
        Returns (answer, chunks) same as query().
        """
        log.info(f"\n🔍 Query (with history): {question}")
        q_embedding = self.embedder.embed_one(question)
        chunks = self.store.query(q_embedding, source_filter=source_filter)

        if not chunks:
            return (
                "⚠️ No relevant content found. Make sure you have ingested a PDF first, "
                "or try rephrasing your question.",
                [],
            )

        log.info(f"   → Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']})")
        answer = self.llm.generate_with_history(question, chunks, history)
        return answer, chunks


# ─────────────────────────────────────────────
# MAIN — CLI usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGPipeline()

    if len(sys.argv) > 1:
        pdf_path = unquote(sys.argv[1])
        pdf_path = os.path.normpath(pdf_path)
        clear = "--clear" in sys.argv

        if not os.path.isfile(pdf_path):
            log.error(f"❌ File not found: {pdf_path}")
            sys.exit(1)

        rag.ingest(pdf_path, clear_existing=clear)

    print("\n💬 Chat started. Type 'exit' or 'quit' to stop.")
    print("   Tip: Type 'stats' to check how many vectors are stored.\n")

    cli_history = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("👋 Exiting.")
            break
        if question.lower() == "stats":
            rag.store.verify()
            continue

        answer, _ = rag.query_with_history(question, history=cli_history)
        print(f"\n🤖 Answer:\n{answer}\n")

        # Keep CLI history updated
        cli_history.append({"role": "user", "content": question})
        cli_history.append({"role": "assistant", "content": answer})