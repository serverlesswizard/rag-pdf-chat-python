import io
import os
import sys
import time
import uuid
from typing import List

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
# LOAD ENV
# ─────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY  = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY")
TESSERACT_CMD     = os.environ.get("TESSERACT_CMD")

# Validate credentials
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing. Please add it to your .env file.")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is missing. Please add it to your .env file.")

# Set Tesseract path if provided
if OCR_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INDEX_NAME        = "pdf-rag-index"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"  # 384-dim, fast & accurate
EMBEDDING_DIM     = 384
CHUNK_SIZE        = 500    # characters per chunk
CHUNK_OVERLAP     = 50     # overlap between chunks
TOP_K             = 5      # how many chunks to retrieve
MIN_CHUNK_LEN     = 20     # minimum characters to keep a chunk
MAX_CONTEXT_CHARS = 12000  # safety cap for LLM context window
GROQ_MODEL        = "llama-3.3-70b-versatile"  # fast & free on Groq


# ─────────────────────────────────────────────
# 1. PDF LOADING & CHUNKING
# ─────────────────────────────────────────────
def load_pdf(pdf_path: str) -> List[dict]:
    """
    Extract text from a PDF file, page by page.
    - Text-based PDFs  → uses PyMuPDF get_text() directly.
    - Scanned/image PDFs → falls back to OCR via pytesseract.
    Returns a list of dicts: { text, page }.
    """
    doc = fitz.open(pdf_path)
    pages = []
    scanned_pages = 0

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if text:
            pages.append({"text": text, "page": i + 1})
        else:
            scanned_pages += 1
            if OCR_AVAILABLE:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        print(f"   🔍 OCR applied on page {i + 1}")
                        pages.append({"text": ocr_text, "page": i + 1})
                    else:
                        print(f"   ⚠️  Page {i + 1}: OCR found no text (blank or unreadable image).")
                except Exception as e:
                    print(f"   ❌ OCR failed on page {i + 1}: {e}")
            else:
                print(f"   ⚠️  Page {i + 1} has no selectable text and OCR is not installed (skipping).")

    doc.close()

    if scanned_pages > 0 and not OCR_AVAILABLE:
        print(f"\n⚠️  {scanned_pages} page(s) were image-based and could not be read.")
        print("   To fix this, follow these steps:")
        print("   1. pip install pytesseract pillow")
        print("   2. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   3. Add to your .env: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n")

    return pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks, filtering out very short ones."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) > MIN_CHUNK_LEN:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_pages(pages: List[dict], **kwargs) -> List[dict]:
    """
    Chunk each page's text and carry page number into every chunk record.
    Returns a list of dicts: { text, page, chunk_index }.
    """
    records = []
    for page in pages:
        page_chunks = chunk_text(page["text"], **kwargs)
        for i, chunk in enumerate(page_chunks):
            records.append({
                "text": chunk,
                "page": page["page"],
                "chunk_index": i,
            })
    return records


# ─────────────────────────────────────────────
# 2. EMBEDDINGS (HuggingFace sentence-transformers)
# ─────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_one(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.model.encode([text]).tolist()[0]


# ─────────────────────────────────────────────
# 3. PINECONE VECTOR STORE
# ─────────────────────────────────────────────
class PineconeStore:
    def __init__(self):
        pc = Pinecone(api_key=PINECONE_API_KEY)

        existing = [i.name for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            print(f"Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("Waiting for index to be ready...")
            while not pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)
            print("✅ Index is ready.")
        else:
            print(f"Using existing Pinecone index: {INDEX_NAME}")

        self.index = pc.Index(INDEX_NAME)

    def upsert(self, records: List[dict], embeddings: List[List[float]], source: str = "pdf"):
        """Upload chunk records + embeddings to Pinecone."""
        vectors = []
        for record, emb in zip(records, embeddings):
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": emb,
                "metadata": {
                    "text": record["text"],
                    "source": source,
                    "page": record["page"],
                    "chunk_index": record["chunk_index"],
                },
            })
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i : i + 100])
        print(f"Upserted {len(vectors)} chunks to Pinecone.")

    def query(self, query_embedding: List[float], top_k: int = TOP_K) -> List[str]:
        """Retrieve top-k relevant chunk texts."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        return [match["metadata"]["text"] for match in results["matches"]]


# ─────────────────────────────────────────────
# 4. GROQ LLM
# ─────────────────────────────────────────────
class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        context = context[:MAX_CONTEXT_CHARS]  # Guard against context overflow

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the answer is not in the context, say so clearly."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────
# 5. RAG PIPELINE
# ─────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store    = PineconeStore()
        self.llm      = GroqLLM()

    def ingest(self, pdf_path: str):
        """Load a PDF, chunk it with page metadata, embed, and store in Pinecone."""
        print(f"\n📄 Ingesting: {pdf_path}")
        pages = load_pdf(pdf_path)

        if not pages:
            print("❌ No text could be extracted from the PDF. Aborting ingestion.")
            print("   Make sure the PDF has selectable text, or install OCR support.")
            return

        records = chunk_pages(pages)
        print(f"   → {len(records)} chunks created across {len(pages)} pages")

        texts      = [r["text"] for r in records]
        embeddings = self.embedder.embed(texts)
        self.store.upsert(records, embeddings, source=os.path.basename(pdf_path))
        print("   ✅ Ingestion complete.\n")

    def query(self, question: str) -> str:
        """Answer a question using retrieved context from Pinecone."""
        print(f"\n🔍 Query: {question}")
        q_embedding = self.embedder.embed_one(question)
        chunks      = self.store.query(q_embedding)

        if not chunks:
            return "⚠️ No relevant content found. Make sure you have ingested a PDF first."

        print(f"   → Retrieved {len(chunks)} chunks")
        answer = self.llm.generate(question, chunks)
        return answer


# ─────────────────────────────────────────────
# MAIN — CLI usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGPipeline()

    # Pass a PDF path as a CLI argument to ingest it
    # Usage: python rag_pipeline.py path/to/document.pdf
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.isfile(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            sys.exit(1)
        rag.ingest(pdf_path)

    # Ask questions in a loop until the user types 'exit' or 'quit'
    print("\n💬 Chat started. Type 'exit' or 'quit' to stop.\n")
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

        answer = rag.query(question)
        print(f"\n🤖 Answer:\n{answer}\n")