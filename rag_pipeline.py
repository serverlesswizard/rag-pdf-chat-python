import io
import os
import sys
import uuid
import logging
import warnings

from typing import List, Optional
from urllib.parse import unquote

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import fitz  # PyMuPDF

from utils import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_PERSIST_DIR,
    EXTRACTED_IMAGES_DIR,
    ensure_directories,
    save_uploaded_file,
)

# Optional OCR support
try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ============================================================
# LOGGING
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="Accessing `__path__`",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

log = logging.getLogger(__name__)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

ensure_directories()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")


# ============================================================
# LLM CONFIGURATION
# ============================================================

# Available providers:
#
#   LLM_PROVIDER=groq
#       → Online Groq API
#
#   LLM_PROVIDER=ollama
#       → Local/offline Ollama
#

LLM_PROVIDER = os.environ.get(
    "LLM_PROVIDER",
    "groq",
).lower()


# ------------------------------------------------------------
# Groq
# ------------------------------------------------------------

GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# ------------------------------------------------------------
# Ollama
# ------------------------------------------------------------

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b",
)

OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)


# ------------------------------------------------------------
# Validate provider
# ------------------------------------------------------------

if LLM_PROVIDER not in (
    "groq",
    "ollama",
):
    raise ValueError(
        f"❌ Invalid LLM_PROVIDER='{LLM_PROVIDER}'. "
        "Use 'groq' or 'ollama'."
    )


# ------------------------------------------------------------
# Validate Groq API key only when Groq is selected
# ------------------------------------------------------------

if (
    LLM_PROVIDER == "groq"
    and not GROQ_API_KEY
):
    raise ValueError(
        "❌ GROQ_API_KEY is missing.\n"
        "Please add it to your .env file."
    )


# ------------------------------------------------------------
# Tesseract
# ------------------------------------------------------------

if OCR_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_CMD
    )


# ============================================================
# RAG CONFIGURATION
# ============================================================

COLLECTION_NAME = "pdf_rag"

# Hugging Face embedding model
EMBEDDING_MODEL = (
    "BAAI/bge-base-en-v1.5"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 8

MIN_CHUNK_LEN = 40

MAX_CONTEXT_CHARS = 14000

UPSERT_BATCH_SIZE = 100

MIN_IMAGE_PIXELS = 100 * 100

MIN_IMAGE_OCR_LEN = 20

SAVE_EXTRACTED_IMAGES = True


if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"❌ CHUNK_OVERLAP ({CHUNK_OVERLAP}) "
        f"must be less than CHUNK_SIZE "
        f"({CHUNK_SIZE})."
    )


# ============================================================
# OCR HELPERS
# ============================================================

def _ocr_image_bytes(
    img_bytes: bytes,
    label: str,
) -> str:
    """
    Run Tesseract OCR on raw image bytes.

    Returns:
        Extracted text or empty string.
    """

    if not OCR_AVAILABLE:
        return ""

    try:
        img = Image.open(
            io.BytesIO(img_bytes)
        )

        text = (
            pytesseract
            .image_to_string(img)
            .strip()
        )

        if len(text) >= MIN_IMAGE_OCR_LEN:
            return text

    except Exception as e:
        log.warning(
            f"   ⚠️ OCR failed on "
            f"{label}: {e}"
        )

    return ""


def _extract_embedded_image_texts(
    page: fitz.Page,
    page_num: int,
    image_save_dir: Optional[str] = None,
) -> List[str]:
    """
    Extract and OCR embedded images
    from a PDF page.
    """

    if not OCR_AVAILABLE:
        return []

    texts = []

    for img_index, img_info in enumerate(
        page.get_images(full=True)
    ):

        xref = img_info[0]
        width = img_info[2]
        height = img_info[3]

        # Ignore tiny images
        if width * height < MIN_IMAGE_PIXELS:
            continue

        try:

            base_image = (
                page.parent.extract_image(
                    xref
                )
            )

            # ------------------------------------------------
            # Save extracted image
            # ------------------------------------------------

            if image_save_dir:

                try:

                    ext = base_image.get(
                        "ext",
                        "png",
                    )

                    img_filename = (
                        f"page{page_num}_"
                        f"img{img_index + 1}."
                        f"{ext}"
                    )

                    image_path = os.path.join(
                        image_save_dir,
                        img_filename,
                    )

                    with open(
                        image_path,
                        "wb",
                    ) as f:

                        f.write(
                            base_image["image"]
                        )

                except Exception as e:

                    log.warning(
                        f"   ⚠️ Could not save "
                        f"image {img_index + 1} "
                        f"on page {page_num}: {e}"
                    )

            # ------------------------------------------------
            # OCR image
            # ------------------------------------------------

            ocr_text = _ocr_image_bytes(
                base_image["image"],
                (
                    f"image "
                    f"{img_index + 1} "
                    f"on page "
                    f"{page_num}"
                ),
            )

            if ocr_text:

                log.info(
                    f"   🖼️ OCR extracted "
                    f"{len(ocr_text)} chars "
                    f"from image "
                    f"{img_index + 1} "
                    f"on page "
                    f"{page_num}"
                )

                texts.append(
                    ocr_text
                )

        except Exception as e:

            log.warning(
                f"   ⚠️ Could not extract "
                f"image {img_index + 1} "
                f"on page {page_num}: {e}"
            )

    return texts


# ============================================================
# PDF LOADING
# ============================================================

def load_pdf(
    pdf_path: str,
    save_extracted_images: bool = (
        SAVE_EXTRACTED_IMAGES
    ),
) -> List[dict]:
    """
    Extract PDF text page-by-page.

    Normal PDF:
        PyMuPDF

    Images/scanned pages:
        Tesseract OCR
    """

    pdf_path = unquote(
        pdf_path
    )

    pdf_path = os.path.normpath(
        pdf_path
    )

    if not os.path.isfile(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    image_save_dir = None

    # --------------------------------------------------------
    # Image storage directory
    # --------------------------------------------------------

    if (
        save_extracted_images
        and OCR_AVAILABLE
    ):

        source_stem = os.path.splitext(
            os.path.basename(pdf_path)
        )[0]

        image_save_dir = os.path.join(
            EXTRACTED_IMAGES_DIR,
            source_stem,
        )

        os.makedirs(
            image_save_dir,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    doc = fitz.open(
        pdf_path
    )

    pages: List[dict] = []

    scanned_pages = 0

    total_embedded_images = 0

    # --------------------------------------------------------
    # Process pages
    # --------------------------------------------------------

    for i, page in enumerate(doc):

        page_num = i + 1

        # Normal selectable text
        page_text = (
            page.get_text()
            .strip()
        )

        # Embedded image OCR
        image_texts = (
            _extract_embedded_image_texts(
                page,
                page_num,
                image_save_dir,
            )
        )

        total_embedded_images += (
            len(image_texts)
        )

        # ====================================================
        # Normal text page
        # ====================================================

        if page_text:

            combined = page_text

            if image_texts:

                combined += (
                    "\n\n"
                    "[Image content on "
                    "this page]:\n"
                    + "\n\n".join(
                        image_texts
                    )
                )

            pages.append(
                {
                    "text": combined,
                    "page": page_num,
                }
            )

        # ====================================================
        # Scanned/image-only page
        # ====================================================

        else:

            scanned_pages += 1

            if OCR_AVAILABLE:

                try:

                    pix = page.get_pixmap(
                        dpi=200
                    )

                    ocr_text = (
                        _ocr_image_bytes(
                            pix.tobytes(
                                "png"
                            ),
                            (
                                f"full page "
                                f"{page_num}"
                            ),
                        )
                    )

                    all_text_parts = (
                        [ocr_text]
                        if ocr_text
                        else []
                    ) + image_texts

                    combined = (
                        "\n\n".join(
                            all_text_parts
                        ).strip()
                    )

                    if combined:

                        log.info(
                            f"   🔍 OCR applied "
                            f"on full page "
                            f"{page_num}"
                        )

                        pages.append(
                            {
                                "text": combined,
                                "page": page_num,
                            }
                        )

                    else:

                        log.warning(
                            f"   ⚠️ Page "
                            f"{page_num}: OCR "
                            f"found no text."
                        )

                except Exception as e:

                    log.error(
                        f"   ❌ OCR failed "
                        f"on page "
                        f"{page_num}: {e}"
                    )

            else:

                log.warning(
                    f"   ⚠️ Page "
                    f"{page_num} has no "
                    f"selectable text "
                    f"and OCR is "
                    f"unavailable."
                )

    doc.close()

    # ========================================================
    # Logging
    # ========================================================

    if (
        scanned_pages > 0
        and not OCR_AVAILABLE
    ):

        log.warning(
            f"\n⚠️ {scanned_pages} "
            f"page(s) were image-based "
            f"and skipped.\n"
            "   Install OCR support with:\n"
            "   pip install pytesseract pillow\n"
        )

    if total_embedded_images > 0:

        log.info(
            f"   🖼️ OCR processed "
            f"{total_embedded_images} "
            f"embedded image(s)."
        )

    log.info(
        f"   ✅ Extracted text from "
        f"{len(pages)} page(s) "
        f"({scanned_pages} image-only pages)"
    )

    return pages


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:

    if overlap >= chunk_size:

        raise ValueError(
            f"overlap ({overlap}) must be "
            f"less than chunk_size "
            f"({chunk_size})"
        )

    chunks: List[str] = []

    start = 0

    text_len = len(text)

    while start < text_len:

        end = min(
            start + chunk_size,
            text_len,
        )

        chunk = (
            text[start:end]
            .strip()
        )

        if len(chunk) >= MIN_CHUNK_LEN:

            chunks.append(
                chunk
            )

        if end >= text_len:
            break

        start += (
            chunk_size - overlap
        )

    return chunks


def chunk_pages(
    pages: List[dict],
    **kwargs,
) -> List[dict]:

    records: List[dict] = []

    for page in pages:

        page_chunks = chunk_text(
            page["text"],
            **kwargs,
        )

        for i, chunk in enumerate(
            page_chunks
        ):

            records.append(
                {
                    "text": chunk,
                    "page": page["page"],
                    "chunk_index": i,
                }
            )

    return records


# ============================================================
# EMBEDDINGS
# ============================================================

class EmbeddingModel:

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):

        log.info(
            f"🧠 Loading embedding model: "
            f"{model_name}"
        )

        self.model = (
            SentenceTransformer(
                model_name
            )
        )

        log.info(
            "✅ Embedding model loaded."
        )

    def embed(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        return (
            self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=64,
            ).tolist()
        )

    def embed_one(
        self,
        text: str,
    ) -> List[float]:

        return (
            self.model.encode(
                [text]
            ).tolist()[0]
        )


# ============================================================
# CHROMADB
# ============================================================

class ChromaStore:
    """
    Local persistent vector database.
    """

    def __init__(self):

        log.info(
            f"🗄️ Initializing ChromaDB: "
            f"{CHROMA_PERSIST_DIR}"
        )

        self.client = (
            chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR
            )
        )

        self.collection = (
            self.client
            .get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                },
            )
        )

        log.info(
            f"✅ ChromaDB ready. "
            f"Collection "
            f"'{COLLECTION_NAME}' "
            f"contains "
            f"{self.collection.count()} "
            f"vectors."
        )

    # --------------------------------------------------------
    # Insert vectors
    # --------------------------------------------------------

    def upsert(
        self,
        records: List[dict],
        embeddings: List[List[float]],
        source: str = "pdf",
    ) -> int:

        ids = [
            str(uuid.uuid4())
            for _ in records
        ]

        metadatas = [
            {
                "text": record["text"],
                "source": source,
                "page": record["page"],
                "chunk_index": record[
                    "chunk_index"
                ],
            }
            for record in records
        ]

        total_upserted = 0

        total_batches = (
            len(ids)
            + UPSERT_BATCH_SIZE
            - 1
        ) // UPSERT_BATCH_SIZE

        for batch_num, i in enumerate(
            range(
                0,
                len(ids),
                UPSERT_BATCH_SIZE,
            ),
            start=1,
        ):

            batch_ids = ids[
                i:i + UPSERT_BATCH_SIZE
            ]

            batch_embeddings = (
                embeddings[
                    i:i + UPSERT_BATCH_SIZE
                ]
            )

            batch_metadatas = (
                metadatas[
                    i:i + UPSERT_BATCH_SIZE
                ]
            )

            try:

                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )

                total_upserted += len(
                    batch_ids
                )

                log.info(
                    f"   Batch "
                    f"{batch_num}/"
                    f"{total_batches}: "
                    f"{len(batch_ids)} "
                    f"vectors ✅"
                )

            except Exception as e:

                log.error(
                    f"   ❌ Batch "
                    f"{batch_num}/"
                    f"{total_batches} "
                    f"failed: {e}"
                )

        log.info(
            f"✅ Total upserted: "
            f"{total_upserted}/"
            f"{len(ids)} vectors."
        )

        return total_upserted

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def query(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K,
        source_filter: Optional[str] = None,
    ) -> List[dict]:

        collection_count = (
            self.collection.count()
        )

        if collection_count == 0:
            return []

        where = (
            {
                "source": {
                    "$eq": source_filter
                }
            }
            if source_filter
            else None
        )

        results = (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=min(
                    top_k,
                    collection_count,
                ),
                include=[
                    "metadatas",
                    "distances",
                ],
                where=where,
            )
        )

        chunks = []

        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):

            score = round(
                1 - distance,
                4,
            )

            chunks.append(
                {
                    "text": metadata[
                        "text"
                    ],
                    "source": metadata.get(
                        "source",
                        "unknown",
                    ),
                    "page": metadata.get(
                        "page",
                        "?",
                    ),
                    "score": score,
                }
            )

        return chunks

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    def verify(self) -> int:

        total = (
            self.collection.count()
        )

        log.info(
            f"📊 Collection "
            f"'{COLLECTION_NAME}': "
            f"{total} vectors"
        )

        return total

    # --------------------------------------------------------
    # Delete source
    # --------------------------------------------------------

    def delete_by_source(
        self,
        source: str,
    ):

        try:

            self.collection.delete(
                where={
                    "source": {
                        "$eq": source
                    }
                }
            )

            log.info(
                f"🗑️ Deleted vectors "
                f"for '{source}'"
            )

        except Exception as e:

            log.error(
                f"❌ Failed to delete "
                f"vectors for "
                f"'{source}': {e}"
            )

    # --------------------------------------------------------
    # Delete everything
    # --------------------------------------------------------

    def delete_all(self):

        try:

            self.client.delete_collection(
                COLLECTION_NAME
            )

            self.collection = (
                self.client
                .get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={
                        "hnsw:space": "cosine"
                    },
                )
            )

            log.info(
                "🗑️ All vectors deleted."
            )

        except Exception as e:

            log.error(
                f"❌ Failed to delete "
                f"collection: {e}"
            )

    # --------------------------------------------------------
    # List PDFs
    # --------------------------------------------------------

    def list_sources(
        self,
    ) -> List[str]:

        if (
            self.collection.count()
            == 0
        ):
            return []

        results = (
            self.collection.get(
                include=["metadatas"]
            )
        )

        sources = list(
            {
                m.get(
                    "source",
                    "unknown",
                )
                for m in results[
                    "metadatas"
                ]
            }
        )

        return sorted(
            sources
        )


# ============================================================
# BASE LLM
# ============================================================

class BaseLLM:
    """
    Shared context builder for Groq
    and Ollama.
    """

    def _build_context(
        self,
        context_chunks: List[dict],
    ) -> str:

        parts = []

        for chunk in context_chunks:

            header = (
                f"[Source: "
                f"{chunk['source']} | "
                f"Page "
                f"{chunk['page']} | "
                f"Score: "
                f"{chunk['score']}]"
            )

            parts.append(
                f"{header}\n"
                f"{chunk['text']}"
            )

        return (
            "\n\n---\n\n".join(
                parts
            )[:MAX_CONTEXT_CHARS]
        )


# ============================================================
# GROQ LLM
# ============================================================

class GroqLLM(BaseLLM):

    def __init__(self):

        if not GROQ_API_KEY:

            raise ValueError(
                "❌ GROQ_API_KEY is missing."
            )

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        self.model = GROQ_MODEL

        log.info(
            f"☁️ Groq initialized: "
            f"{self.model}"
        )

    # --------------------------------------------------------
    # Normal query
    # --------------------------------------------------------

    def generate(
        self,
        query: str,
        context_chunks: List[dict],
    ) -> str:

        context = (
            self._build_context(
                context_chunks
            )
        )

        system_prompt = (
            "You are a precise and helpful "
            "document assistant. "
            "Answer the user's question using "
            "ONLY the provided document context. "
            "Do not use outside knowledge. "
            "Do not guess or hallucinate. "
            "If the answer cannot be found in "
            "the context, clearly say that the "
            "information is not available "
            "in the document. "
            "When relevant, mention the source "
            "and page number."
        )

        try:

            response = (
                self.client
                .chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                system_prompt
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n"
                                f"{context}\n\n"
                                f"Question: "
                                f"{query}"
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
            )

            return (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            log.error(
                f"❌ Groq API error: {e}"
            )

            return (
                f"❌ Error generating "
                f"response: {e}"
            )

    # --------------------------------------------------------
    # Query with history
    # --------------------------------------------------------

    def generate_with_history(
        self,
        query: str,
        context_chunks: List[dict],
        history: Optional[
            List[dict]
        ] = None,
    ) -> str:

        context = (
            self._build_context(
                context_chunks
            )
        )

        system_prompt = (
            "You are a precise and helpful "
            "document assistant. "
            "Answer the user's question using "
            "ONLY the provided document context. "
            "Conversation history may be used "
            "to understand follow-up questions, "
            "but answers must remain grounded "
            "in the document context. "
            "Do not guess or hallucinate. "
            "If the answer cannot be found, "
            "clearly say so. "
            "When relevant, mention the source "
            "and page number.\n\n"
            f"Context:\n{context}"
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history:

            messages.extend(
                history[-10:]
            )

        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        try:

            response = (
                self.client
                .chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024,
                )
            )

            return (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            log.error(
                f"❌ Groq API error: {e}"
            )

            return (
                f"❌ Error generating "
                f"response: {e}"
            )


# ============================================================
# OLLAMA LLM
# ============================================================

class OllamaLLM(BaseLLM):
    """
    Local Ollama backend.

    Once the model is downloaded,
    inference works without internet.
    """

    def __init__(self):

        # Import Ollama only if this backend
        # is actually selected.
        try:

            import ollama

        except ImportError:

            raise ImportError(
                "❌ Ollama Python package "
                "is not installed.\n"
                "Install it inside the venv with:\n"
                "pip install ollama"
            )

        self.client = ollama.Client(
            host=OLLAMA_HOST
        )

        self.model = OLLAMA_MODEL

        log.info(
            f"🦙 Ollama initialized: "
            f"{self.model}"
        )

        log.info(
            f"   Ollama host: "
            f"{OLLAMA_HOST}"
        )

    # --------------------------------------------------------
    # Normal query
    # --------------------------------------------------------

    def generate(
        self,
        query: str,
        context_chunks: List[dict],
    ) -> str:

        context = (
            self._build_context(
                context_chunks
            )
        )

        system_prompt = (
            "You are a precise and helpful "
            "document assistant. "
            "Answer the user's question using "
            "ONLY the provided document context. "
            "Do not use outside knowledge. "
            "Do not guess or hallucinate. "
            "If the answer cannot be found in "
            "the context, clearly say that the "
            "information is not available "
            "in the document. "
            "When relevant, mention the source "
            "and page number."
        )

        try:

            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n"
                            f"{context}\n\n"
                            f"Question: "
                            f"{query}"
                        ),
                    },
                ],
                options={
                    "temperature": 0.2,
                },
            )

            return response[
                "message"
            ][
                "content"
            ]

        except Exception as e:

            log.error(
                f"❌ Ollama error: {e}"
            )

            return (
                f"❌ Error generating "
                f"response: {e}"
            )

    # --------------------------------------------------------
    # Query with history
    # --------------------------------------------------------

    def generate_with_history(
        self,
        query: str,
        context_chunks: List[dict],
        history: Optional[
            List[dict]
        ] = None,
    ) -> str:

        context = (
            self._build_context(
                context_chunks
            )
        )

        system_prompt = (
            "You are a precise and helpful "
            "document assistant. "
            "Answer the user's question using "
            "ONLY the provided document context. "
            "You may use conversation history "
            "to understand follow-up questions, "
            "but answers must remain grounded "
            "in the document context. "
            "Do not guess or hallucinate. "
            "If the answer cannot be found, "
            "clearly say so. "
            "When relevant, mention the source "
            "and page number.\n\n"
            f"Context:\n{context}"
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history:

            messages.extend(
                history[-10:]
            )

        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        try:

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.2,
                },
            )

            return response[
                "message"
            ][
                "content"
            ]

        except Exception as e:

            log.error(
                f"❌ Ollama error: {e}"
            )

            return (
                f"❌ Error generating "
                f"response: {e}"
            )


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:

    def __init__(self):

        log.info(
            "====================================="
        )

        log.info(
            "🚀 Initializing RAG Pipeline"
        )

        log.info(
            f"🤖 LLM Provider: "
            f"{LLM_PROVIDER.upper()}"
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.embedder = (
            EmbeddingModel()
        )

        # ----------------------------------------------------
        # Vector database
        # ----------------------------------------------------

        self.store = ChromaStore()

        # ----------------------------------------------------
        # Select LLM
        # ----------------------------------------------------

        if LLM_PROVIDER == "groq":

            self.llm = GroqLLM()

        elif LLM_PROVIDER == "ollama":

            self.llm = OllamaLLM()

        else:

            raise ValueError(
                f"Unsupported LLM provider: "
                f"{LLM_PROVIDER}"
            )

        log.info(
            "====================================="
        )

    # ========================================================
    # INGEST PDF
    # ========================================================

    def ingest(
        self,
        pdf_path: str,
        clear_existing: bool = False,
    ):

        log.info(
            f"\n📥 Ingesting: "
            f"{pdf_path}"
        )

        source_name = (
            os.path.basename(
                pdf_path
            )
        )

        # ----------------------------------------------------
        # Clear existing vectors
        # ----------------------------------------------------

        if clear_existing:

            log.info(
                f"🗑️ Removing existing "
                f"vectors for "
                f"'{source_name}'..."
            )

            self.store.delete_by_source(
                source_name
            )

        # ----------------------------------------------------
        # Extract PDF
        # ----------------------------------------------------

        pages = load_pdf(
            pdf_path
        )

        if not pages:

            log.error(
                "❌ No text extracted "
                "from PDF."
            )

            return

        # ----------------------------------------------------
        # Chunk
        # ----------------------------------------------------

        records = chunk_pages(
            pages
        )

        log.info(
            f"   ✂️ Created "
            f"{len(records)} chunks "
            f"from "
            f"{len(pages)} pages"
        )

        if not records:

            log.error(
                "❌ No valid chunks "
                "created."
            )

            return

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        texts = [
            record["text"]
            for record in records
        ]

        log.info(
            f"   🧮 Generating "
            f"embeddings for "
            f"{len(texts)} chunks..."
        )

        embeddings = (
            self.embedder.embed(
                texts
            )
        )

        # ----------------------------------------------------
        # Store vectors
        # ----------------------------------------------------

        self.store.upsert(
            records,
            embeddings,
            source=source_name,
        )

        stored = (
            self.store.verify()
        )

        log.info(
            f"   ✅ Ingestion complete. "
            f"{stored} total vectors."
        )

    # ========================================================
    # STREAMLIT UPLOAD
    # ========================================================

    def ingest_uploaded_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        clear_existing: bool = False,
    ) -> str:

        saved_path = (
            save_uploaded_file(
                KNOWLEDGE_BASE_DIR,
                file_bytes,
                filename,
            )
        )

        self.ingest(
            saved_path,
            clear_existing=clear_existing,
        )

        return saved_path

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
    ) -> tuple[
        str,
        List[dict]
    ]:

        log.info(
            f"\n❓ Query: "
            f"{question}"
        )

        # ----------------------------------------------------
        # Question → embedding
        # ----------------------------------------------------

        q_embedding = (
            self.embedder.embed_one(
                question
            )
        )

        # ----------------------------------------------------
        # Search ChromaDB
        # ----------------------------------------------------

        chunks = (
            self.store.query(
                q_embedding,
                source_filter=source_filter,
            )
        )

        if not chunks:

            return (
                "⚠️ No relevant content "
                "found. Make sure you "
                "have ingested a PDF first.",
                [],
            )

        log.info(
            f"   ✅ Retrieved "
            f"{len(chunks)} chunks "
            f"(top score: "
            f"{chunks[0]['score']})"
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        answer = (
            self.llm.generate(
                question,
                chunks,
            )
        )

        return answer, chunks

    # ========================================================
    # QUERY WITH HISTORY
    # ========================================================

    def query_with_history(
        self,
        question: str,
        history: Optional[
            List[dict]
        ] = None,
        source_filter: Optional[str] = None,
    ) -> tuple[
        str,
        List[dict]
    ]:

        log.info(
            f"\n❓ Query "
            f"(with history): "
            f"{question}"
        )

        q_embedding = (
            self.embedder.embed_one(
                question
            )
        )

        chunks = (
            self.store.query(
                q_embedding,
                source_filter=source_filter,
            )
        )

        if not chunks:

            return (
                "⚠️ No relevant content "
                "found.",
                [],
            )

        log.info(
            f"   ✅ Retrieved "
            f"{len(chunks)} chunks "
            f"(top score: "
            f"{chunks[0]['score']})"
        )

        answer = (
            self.llm
            .generate_with_history(
                question,
                chunks,
                history,
            )
        )

        return answer, chunks


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    log.info(
        "====================================="
    )

    log.info(
        "🚀 Starting RAG Pipeline"
    )

    log.info(
        f"🤖 LLM Provider: "
        f"{LLM_PROVIDER.upper()}"
    )

    if LLM_PROVIDER == "groq":

        log.info(
            f"☁️ Groq Model: "
            f"{GROQ_MODEL}"
        )

    elif LLM_PROVIDER == "ollama":

        log.info(
            f"🦙 Ollama Model: "
            f"{OLLAMA_MODEL}"
        )

        log.info(
            f"   Ollama Host: "
            f"{OLLAMA_HOST}"
        )

    log.info(
        "====================================="
    )

    # --------------------------------------------------------
    # Initialize RAG
    # --------------------------------------------------------

    rag = RAGPipeline()

    # --------------------------------------------------------
    # Optional PDF ingestion
    #
    # python rag_pipeline.py document.pdf
    #
    # python rag_pipeline.py document.pdf --clear
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        pdf_path = unquote(
            sys.argv[1]
        )

        pdf_path = os.path.normpath(
            pdf_path
        )

        clear = (
            "--clear"
            in sys.argv
        )

        if not os.path.isfile(
            pdf_path
        ):

            log.error(
                f"❌ File not found: "
                f"{pdf_path}"
            )

            log.error(
                "Tip: Wrap the path "
                "in quotes if it "
                "contains spaces."
            )

            sys.exit(1)

        rag.ingest(
            pdf_path,
            clear_existing=clear,
        )

    # --------------------------------------------------------
    # Chat
    # --------------------------------------------------------

    print(
        "\n💬 Chat started."
    )

    print(
        "Type 'exit' or 'quit' "
        "to stop."
    )

    print(
        "Commands:"
    )

    print(
        "  stats   → vector count"
    )

    print(
        "  sources → list indexed PDFs"
    )

    print()

    cli_history: List[
        dict
    ] = []

    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\n👋 Exiting."
            )

            break

        if not question:
            continue

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() in (
            "exit",
            "quit",
        ):

            print(
                "👋 Exiting."
            )

            break

        # ----------------------------------------------------
        # Stats
        # ----------------------------------------------------

        if question.lower() == "stats":

            rag.store.verify()

            continue

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if question.lower() == "sources":

            sources = (
                rag.store.list_sources()
            )

            print(
                f"   Indexed PDFs: "
                f"{sources or 'none'}"
            )

            continue

        # ----------------------------------------------------
        # Query
        # ----------------------------------------------------

        answer, _ = (
            rag.query_with_history(
                question,
                history=cli_history,
            )
        )

        print(
            f"\n🤖 Answer:\n"
            f"{answer}\n"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        cli_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        cli_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )