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
import chromadb
from groq import Groq
import fitz  # PyMuPDF

# Optional OCR support for scanned PDFs
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------------------------
# LOGGING SETUP
# ---------------------------------------------
import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ---------------------------------------------
# LOAD ENV
# ---------------------------------------------
load_dotenv()

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY")
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")

if not GROQ_API_KEY:
    raise ValueError("? GROQ_API_KEY is missing. Please add it to your .env file.")

if OCR_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ---------------------------------------------
# CONFIG
# ---------------------------------------------
CHROMA_PERSIST_DIR = "./chroma_db"       # Local folder where ChromaDB stores data
COLLECTION_NAME    = "pdf_rag"           # ChromaDB collection name
EMBEDDING_MODEL    = "all-MiniLM-L6-v2" # 384-dim, fast & accurate
CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 150
TOP_K              = 8
MIN_CHUNK_LEN      = 40
MAX_CONTEXT_CHARS  = 14000
GROQ_MODEL         = "llama-3.3-70b-versatile"
UPSERT_BATCH_SIZE  = 100
MIN_IMAGE_PIXELS   = 100 * 100
MIN_IMAGE_OCR_LEN  = 20

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"? CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})."
    )


# ---------------------------------------------
# OCR HELPERS
# ---------------------------------------------
def _ocr_image_bytes(img_bytes: bytes, label: str) -> str:
    """Run pytesseract on raw image bytes. Returns stripped OCR text or ''."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img).strip()
        if len(text) >= MIN_IMAGE_OCR_LEN:
            return text
    except Exception as e:
        log.warning(f"   ??  OCR failed on {label}: {e}")
    return ""


def _extract_embedded_image_texts(page: fitz.Page, page_num: int) -> List[str]:
    """Extract and OCR every embedded image on a PDF page. Skips tiny images."""
    if not OCR_AVAILABLE:
        return []

    texts = []
    for img_index, img_info in enumerate(page.get_images(full=True)):
        xref, _, width, height = img_info[0], img_info[1], img_info[2], img_info[3]

        if width * height < MIN_IMAGE_PIXELS:
            continue

        try:
            base_image = page.parent.extract_image(xref)
            ocr_text = _ocr_image_bytes(
                base_image["image"], f"image {img_index + 1} on page {page_num}"
            )
            if ocr_text:
                log.info(f"   ???  OCR extracted {len(ocr_text)} chars from image {img_index + 1} on page {page_num}")
                texts.append(ocr_text)
        except Exception as e:
            log.warning(f"   ??  Could not extract image {img_index + 1} on page {page_num}: {e}")

    return texts


# ---------------------------------------------
# 1. PDF LOADING & CHUNKING
# ---------------------------------------------
def load_pdf(pdf_path: str) -> List[dict]:
    """
    Extract text from a PDF file page by page.
    - Text-based pages  ? PyMuPDF get_text()
    - Image/scanned pages ? pytesseract OCR
    Returns: [{ text, page }, ...]
    """
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

        image_texts = _extract_embedded_image_texts(page, page_num)
        total_embedded_images += len(image_texts)

        if page_text:
            combined = page_text
            if image_texts:
                combined += "\n\n[Image content on this page]:\n" + "\n\n".join(image_texts)
            pages.append({"text": combined, "page": page_num})
        else:
            scanned_pages += 1
            if OCR_AVAILABLE:
                try:
                    pix = page.get_pixmap(dpi=200)
                    ocr_text = _ocr_image_bytes(pix.tobytes("png"), f"full page {page_num}")
                    all_text_parts = ([ocr_text] if ocr_text else []) + image_texts
                    combined = "\n\n".join(all_text_parts).strip()
                    if combined:
                        log.info(f"   ?? OCR applied on full page {page_num}")
                        pages.append({"text": combined, "page": page_num})
                    else:
                        log.warning(f"   ??  Page {page_num}: OCR found no text.")
                except Exception as e:
                    log.error(f"   ? OCR failed on page {page_num}: {e}")
            else:
                log.warning(f"   ??  Page {page_num} has no selectable text and OCR is unavailable.")

    doc.close()

    if scanned_pages > 0 and not OCR_AVAILABLE:
        log.warning(
            f"\n??  {scanned_pages} page(s) were image-based and skipped.\n"
            "   To enable OCR: pip install pytesseract pillow and set TESSERACT_CMD in .env\n"
        )

    if total_embedded_images > 0:
        log.info(f"   ???  OCR processed {total_embedded_images} embedded image(s) across all pages.")

    log.info(f"   ? Extracted text from {len(pages)} page(s) ({scanned_pages} image-only skipped)")
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping fixed-size character chunks."""
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
    """Chunk all pages and attach page metadata to each chunk."""
    records: List[dict] = []
    for page in pages:
        for i, chunk in enumerate(chunk_text(page["text"], **kwargs)):
            records.append({"text": chunk, "page": page["page"], "chunk_index": i})
    return records


# ---------------------------------------------
# 2. EMBEDDINGS
# ---------------------------------------------
class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        log.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True, batch_size=64).tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]


# ---------------------------------------------
# 3. CHROMADB VECTOR STORE
# ---------------------------------------------
class ChromaStore:
    """
    Local vector store using ChromaDB.
    Data is persisted to CHROMA_PERSIST_DIR on disk —
    no API key, no internet, completely free.
    """

    def __init__(self):
        log.info(f"Initializing ChromaDB at: {CHROMA_PERSIST_DIR}")
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # cosine similarity matches the embedding model's metric
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"? ChromaDB ready. Collection '{COLLECTION_NAME}' has {self.collection.count()} vectors.")

    def upsert(self, records: List[dict], embeddings: List[List[float]], source: str = "pdf") -> int:
        """
        Upload chunk records + embeddings to ChromaDB in batches.
        Uses uuid for unique IDs so re-ingesting the same PDF adds new vectors
        rather than overwriting (use delete_by_source() first to avoid duplicates).
        Returns total upserted count.
        """
        ids        = [str(uuid.uuid4()) for _ in records]
        metadatas  = [
            {
                "text":        record["text"],
                "source":      source,
                "page":        record["page"],
                "chunk_index": record["chunk_index"],
            }
            for record in records
        ]

        total_upserted = 0
        total_batches  = (len(ids) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

        for batch_num, i in enumerate(range(0, len(ids), UPSERT_BATCH_SIZE), start=1):
            batch_ids        = ids[i : i + UPSERT_BATCH_SIZE]
            batch_embeddings = embeddings[i : i + UPSERT_BATCH_SIZE]
            batch_metadatas  = metadatas[i : i + UPSERT_BATCH_SIZE]

            try:
                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
                total_upserted += len(batch_ids)
                log.info(f"   Batch {batch_num}/{total_batches}: upserted {len(batch_ids)} vectors ?")
            except Exception as e:
                log.error(f"   ? Batch {batch_num}/{total_batches} failed: {e}")

        log.info(f"? Total upserted: {total_upserted}/{len(ids)} vectors.")
        return total_upserted

    def query(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve top-k relevant chunks by cosine similarity.
        Optionally filter by source filename using ChromaDB's where clause.
        """
        where = {"source": {"$eq": source_filter}} if source_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            include=["metadatas", "distances"],
            where=where,
        )

        chunks = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            # ChromaDB cosine distance ? similarity: score = 1 - distance
            # (distance=0 means identical, distance=2 means opposite)
            score = round(1 - distance, 4)
            chunks.append({
                "text":   metadata["text"],
                "source": metadata.get("source", "unknown"),
                "page":   metadata.get("page", "?"),
                "score":  score,
            })

        return chunks

    def verify(self) -> int:
        """Return the total vector count stored in the collection."""
        total = self.collection.count()
        log.info(f"?? ChromaDB collection '{COLLECTION_NAME}': {total} vectors stored.")
        return total

    def delete_by_source(self, source: str):
        """
        Delete all vectors belonging to a specific source file.
        Use this before re-ingesting a PDF to avoid duplicate chunks.
        """
        try:
            self.collection.delete(where={"source": {"$eq": source}})
            log.info(f"???  Deleted all vectors for source: '{source}'")
        except Exception as e:
            log.error(f"? Failed to delete vectors for '{source}': {e}")

    def delete_all(self):
        """Wipe the entire collection and recreate it."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            log.info("???  All vectors deleted and collection reset.")
        except Exception as e:
            log.error(f"? Failed to delete collection: {e}")

    def list_sources(self) -> List[str]:
        """Return a list of all unique source filenames in the collection."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = list({m.get("source", "unknown") for m in results["metadatas"]})
        return sorted(sources)


# ---------------------------------------------
# 4. GROQ LLM
# ---------------------------------------------
class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def _build_context(self, context_chunks: List[dict]) -> str:
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
            log.error(f"? Groq API error: {e}")
            return f"? Error generating response: {e}"

    def generate_with_history(
        self,
        query: str,
        context_chunks: List[dict],
        history: Optional[List[dict]] = None,
    ) -> str:
        """Multi-turn: includes prior conversation history for follow-up questions."""
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
        if history:
            messages.extend(history[-10:])  # cap at last 10 turns
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
            log.error(f"? Groq API error: {e}")
            return f"? Error generating response: {e}"


# ---------------------------------------------
# 5. RAG PIPELINE
# ---------------------------------------------
class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store    = ChromaStore()
        self.llm      = GroqLLM()

    def ingest(self, pdf_path: str, clear_existing: bool = False):
        """
        Full ingestion pipeline:
        1. Optionally delete existing vectors for this source
        2. Load & extract PDF text (with OCR fallback)
        3. Chunk with overlap
        4. Embed all chunks
        5. Upsert to ChromaDB
        6. Verify stored count
        """
        log.info(f"\n?? Ingesting: {pdf_path}")
        source_name = os.path.basename(pdf_path)

        if clear_existing:
            log.info(f"???  Removing existing vectors for '{source_name}'...")
            self.store.delete_by_source(source_name)

        pages = load_pdf(pdf_path)
        if not pages:
            log.error("? No text extracted from PDF. Aborting.")
            return

        records = chunk_pages(pages)
        log.info(f"   ? {len(records)} chunks from {len(pages)} pages")

        if not records:
            log.error("? No valid chunks created. Check PDF content.")
            return

        texts = [r["text"] for r in records]
        log.info(f"   ? Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)

        upserted = self.store.upsert(records, embeddings, source=source_name)
        stored = self.store.verify()
        log.info(f"   ? Ingestion complete. {stored} total vectors in collection.\n")

    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
    ) -> tuple[str, List[dict]]:
        """Retrieve relevant chunks and return (answer, chunks)."""
        log.info(f"\n?? Query: {question}")
        q_embedding = self.embedder.embed_one(question)
        chunks = self.store.query(q_embedding, source_filter=source_filter)

        if not chunks:
            return (
                "?? No relevant content found. Make sure you have ingested a PDF first, "
                "or try rephrasing your question.",
                [],
            )

        log.info(f"   ? Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']})")
        answer = self.llm.generate(question, chunks)
        return answer, chunks

    def query_with_history(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        source_filter: Optional[str] = None,
    ) -> tuple[str, List[dict]]:
        """Multi-turn variant for follow-up questions. Returns (answer, chunks)."""
        log.info(f"\n?? Query (with history): {question}")
        q_embedding = self.embedder.embed_one(question)
        chunks = self.store.query(q_embedding, source_filter=source_filter)

        if not chunks:
            return (
                "?? No relevant content found. Make sure you have ingested a PDF first, "
                "or try rephrasing your question.",
                [],
            )

        log.info(f"   ? Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']})")
        answer = self.llm.generate_with_history(question, chunks, history)
        return answer, chunks


# ---------------------------------------------
# MAIN — CLI usage
# ---------------------------------------------
if __name__ == "__main__":
    rag = RAGPipeline()

    # Usage:
    #   python rag_pipeline_chroma.py path/to/doc.pdf          # ingest
    #   python rag_pipeline_chroma.py path/to/doc.pdf --clear  # re-ingest (removes old vectors first)
    if len(sys.argv) > 1:
        pdf_path = unquote(sys.argv[1])
        pdf_path = os.path.normpath(pdf_path)
        clear    = "--clear" in sys.argv

        if not os.path.isfile(pdf_path):
            log.error(f"? File not found: {pdf_path}")
            log.error('   Tip: Wrap path in quotes if it contains spaces.')
            sys.exit(1)

        rag.ingest(pdf_path, clear_existing=clear)

    print("\n?? Chat started. Type 'exit' or 'quit' to stop.")
    print("   Commands: 'stats' ? vector count | 'sources' ? list ingested PDFs\n")

    cli_history: List[dict] = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n?? Exiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("?? Exiting.")
            break
        if question.lower() == "stats":
            rag.store.verify()
            continue
        if question.lower() == "sources":
            sources = rag.store.list_sources()
            print(f"   Indexed PDFs: {sources or 'none'}")
            continue

        answer, _ = rag.query_with_history(question, history=cli_history)
        print(f"\n?? Answer:\n{answer}\n")

        cli_history.append({"role": "user",      "content": question})
        cli_history.append({"role": "assistant",  "content": answer})