# 🧠 rag-pdf-chat

Chat with your PDF documents using a RAG (Retrieval-Augmented Generation) pipeline powered by **Groq LLM**, **Pinecone** vector store, and **sentence-transformers** for embeddings.

---

## ✨ Features

- 📄 Ingest any PDF — text-based or scanned
- 🔍 OCR support for scanned/image-based PDFs via Tesseract
- ⚡ Fast responses powered by Groq (Llama 3.3 70B)
- 🧠 Semantic search via Pinecone vector store
- 💬 Continuous chat loop — ask as many questions as you want
- 🔒 Secure API key management via `.env`

---

## 🗂️ Project Structure

```
rag-pdf-chat/
│
├── rag_pipeline.py      # Main RAG pipeline
├── requirements.txt     # Python dependencies
├── .env                 # Your API keys (never commit this)
├── .env.example         # Safe template to share
├── .gitignore           # Ignores .env and cache files
└── README.md            # You are here
```

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | [Groq](https://console.groq.com) — Llama 3.3 70B |
| Vector Store | [Pinecone](https://app.pinecone.io) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| PDF Parsing | PyMuPDF (`fitz`) |
| OCR (scanned PDFs) | Tesseract + pytesseract |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/rag-pdf-chat.git
cd rag-pdf-chat
```

### 2. Create and activate a virtual environment

**Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract (for scanned PDFs only)
Download and install from:
👉 https://github.com/UB-Mannheim/tesseract/wiki

### 5. Set up your `.env` file
Copy the example and fill in your keys:
```bash
cp .env.example .env
```

```env
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

| Key | Where to get it |
|-----|----------------|
| `PINECONE_API_KEY` | https://app.pinecone.io → API Keys |
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `TESSERACT_CMD` | Path to Tesseract install (Windows only) |

---

## 💬 Usage

### Ingest a PDF and start chatting
```bash
python rag_pipeline.py path/to/your/document.pdf
```

### Just chat (PDF already ingested before)
```bash
python rag_pipeline.py
```

### Example session
```
📄 Ingesting: posh-policy.pdf
   🔍 OCR applied on page 1
   → 42 chunks created across 10 pages
   ✅ Ingestion complete.

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

> 💡 You only need to ingest a PDF **once**. The data is stored in Pinecone and can be queried anytime without re-ingesting.

---

## 🔒 Security

- Never commit your `.env` file — it's listed in `.gitignore`
- Use `.env.example` as a safe template to share with others

---

## 📦 Dependencies

```
python-dotenv
sentence-transformers
pinecone
groq
pymupdf
pytesseract
pillow
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `0 chunks created` | PDF is scanned — install Tesseract OCR |
| `cannot import name 'Pinecone'` | Run `pip uninstall pinecone pinecone-client -y && pip install pinecone` |
| `GROQ_API_KEY is missing` | Add your Groq key to `.env` |
| `tesseract is not recognized` | Set `TESSERACT_CMD` in `.env` with the full path |

---

## 📄 License

MIT License — feel free to use and modify.
