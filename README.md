# 🧠 rag-pdf-chat

Chat with your PDF documents using a RAG (Retrieval-Augmented Generation) pipeline powered by **Ollama (Qwen2.5 1.5B)** for fully local LLM inference, **ChromaDB** vector store, and **Hugging Face BGE embeddings** — with a clean **Streamlit** web interface.

---

## ✨ Features

- 📄 Ingest any PDF — text-based or scanned
- 🔍 OCR support for scanned/image-based PDFs via Tesseract
- 🦙 Fully local LLM inference via Ollama (Qwen2.5 1.5B) — no external API needed
- 🧠 Semantic search via ChromaDB vector store
- 🔢 High-quality BGE embeddings (`BAAI/bge-base-en-v1.5`)
- 🌐 Simple interactive Streamlit web interface
- 💬 Natural language question answering
- 🗂️ Persistent vector storage — ingest once, query anytime
- 🔒 Data privacy — everything runs locally on your machine

---

## 🗂️ Project Structure

```
rag-pdf-chat/
│
├── app.py               # Streamlit web interface
├── rag_pipeline.py      # Core RAG pipeline logic
├── requirements.txt     # Python dependencies
├── .env                 # Your config (never commit this)
├── .env.example         # Safe template to share
├── .gitignore           # Ignores .env and cache files
├── chroma_db/           # Persistent vector store (auto-created)
└── README.md            # You are here
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| LLM | [Ollama](https://ollama.com) — Qwen2.5 1.5B (runs locally) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | `BAAI/bge-base-en-v1.5` via Hugging Face |
| PDF Parsing | PyMuPDF (`fitz`) |
| Web UI | Streamlit |
| OCR (scanned PDFs) | Tesseract + pytesseract |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/serverlesswizard/rag-pdf-chat.git
cd rag-pdf-chat
```

### 2. Install and set up Ollama

Download and install Ollama from 👉 https://ollama.com/download

Then pull the Qwen2.5 model:
```bash
ollama pull qwen2.5:1.5b
```

Verify it works:
```bash
ollama run qwen2.5:1.5b "Hello!"
```

### 3. Create and activate a virtual environment

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Install Tesseract (for scanned PDFs only)

**Windows:** Download and install from 👉 https://github.com/UB-Mannheim/tesseract/wiki

**Linux:**
```bash
sudo apt update && sudo apt install tesseract-ocr
```

Verify:
```bash
tesseract --version
```

### 6. Set up your `.env` file
```bash
cp .env.example .env
```

Fill in your config:
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe   # Windows only
HF_TOKEN=your-huggingface-token                               # Optional but recommended
```

| Key | Where to get it |
|---|---|
| `TESSERACT_CMD` | Full path to Tesseract install (Windows only) |
| `HF_TOKEN` | https://huggingface.co/settings/tokens |

> No API keys needed for the LLM — Ollama runs completely locally!

---

## ▶️ Running the App

Make sure Ollama is running in the background, then:

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`) and start chatting with your PDFs.

---

## 💬 Usage

**Step 1 — Upload a PDF** through the Streamlit sidebar.

**Step 2 — Wait for processing.** The app extracts text, chunks it, generates BGE embeddings, and stores them in ChromaDB.

**Step 3 — Ask questions** in natural language:

```
What is this document about?
Summarize the key findings.
What are the important dates mentioned?
Who are the stakeholders involved?
What solutions are proposed?
```

### Example session
```
📄 Ingesting: Posh-Policy.pdf
   → Extracted text from 10 page(s)
   → 42 chunks from 10 pages
   → Embedding 42 chunks with BAAI/bge-base-en-v1.5...
   ✅ Ingestion complete. 42 vectors stored in ChromaDB.

💬 Chat started. Type 'exit' or 'quit' to stop.

You: What is this document about?
🤖 Answer:
This document is about the POSH (Prevention of Sexual Harassment) policy...

You: Who can file a complaint?
🤖 Answer:
Any employee who experiences sexual harassment at the workplace can file...

You: exit
👋 Exiting.
```

> 💡 You only need to ingest a PDF **once**. The data persists in ChromaDB between sessions — no need to re-ingest unless the document changes.

---

## 🎛️ RAG Parameters

These can be tuned in `rag_pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `8` | Chunks retrieved per query |
| `MIN_CHUNK_LEN` | `40` | Minimum chunk length to keep |
| `MAX_CONTEXT_CHARS` | `14000` | Max characters sent to LLM |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Hugging Face embedding model |
| `LLM_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| `COLLECTION_NAME` | `pdf_rag` | ChromaDB collection name |

---

## 📦 Dependencies

```
streamlit>=1.32.0
chromadb>=1.0.0
sentence-transformers>=3.0.0
ollama>=0.1.0
pymupdf>=1.24.0
python-dotenv>=1.0.0
pytesseract>=0.3.10
pillow>=10.0.0
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `0 chunks created` | PDF is scanned — install Tesseract OCR |
| `tesseract is not recognized` | Set `TESSERACT_CMD` in `.env` with the full path |
| `File not found` error | Wrap the path in double quotes: `"C:\Users\Name\file.pdf"` |
| Ollama connection error | Make sure Ollama is running: `ollama serve` |
| `model not found` error | Run `ollama pull qwen2.5:1.5b` to download the model |
| Broken virtual environment | Delete with `rd /s /q venv`, recreate with `python -m venv venv` |
| HuggingFace rate limit warning | Set `HF_TOKEN` in `.env` |
| ChromaDB corrupted / stale data | Delete `chroma_db/` folder and re-ingest |
| Streamlit not found | Make sure venv is activated, then `pip install -r requirements.txt` |

**Reset ChromaDB:**
```cmd
rd /s /q chroma_db
```
Then restart the app and re-ingest your documents.

---

## 🔒 Security & Privacy

- Everything runs **locally** — your documents never leave your machine
- No external LLM API calls — Ollama handles inference on-device
- Never commit your `.env` file — it's listed in `.gitignore`
- Use `.env.example` as a safe template to share with others

---

## 👨‍💻 Author

**Vishal Anand** — Cloud / DevOps Engineer

[![GitHub](https://img.shields.io/badge/GitHub-serverlesswizard-181717?style=flat&logo=github)](https://github.com/serverlesswizard)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vishalanand25-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/vishalanand25)

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
