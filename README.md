Absolutely. Here’s the complete README.md, ready to copy directly into your project.

# 🤖 AI-Powered RAG Document Assistant


An AI-powered **Retrieval-Augmented Generation (RAG) document assistant** that allows users to upload PDF documents and ask questions about their contents using natural language.


The application processes documents, extracts text and images, creates semantic embeddings, stores them in a vector database, retrieves the most relevant information, and uses an LLM to generate context-aware answers.


Built with **Python, Streamlit, ChromaDB, Hugging Face Sentence Transformers, Groq, PyMuPDF, and Tesseract OCR**.

<<<<<<< HEAD
Chat with your PDF documents using a RAG (Retrieval-Augmented Generation) pipeline powered by **Ollama (Qwen2.5 1.5B)** for fully local LLM inference, **ChromaDB** vector store, and **Hugging Face BGE embeddings** — with a clean **Streamlit** web interface.
=======
>>>>>>> e9caee6b84ce17c1cef660306944a07078fda358

---


<<<<<<< HEAD
- 📄 Ingest any PDF — text-based or scanned
- 🔍 OCR support for scanned/image-based PDFs via Tesseract
- 🦙 Fully local LLM inference via Ollama (Qwen2.5 1.5B) — no external API needed
- 🧠 Semantic search via ChromaDB vector store
- 🔢 High-quality BGE embeddings (`BAAI/bge-base-en-v1.5`)
- 🌐 Simple interactive Streamlit web interface
- 💬 Natural language question answering
- 🗂️ Persistent vector storage — ingest once, query anytime
- 🔒 Data privacy — everything runs locally on your machine
=======
## 🚀 Features


- 📄 Upload and analyze PDF documents
- 🔎 Semantic document search using vector embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 💬 Natural-language question answering
- ⚡ Fast LLM responses using Groq API
- 🗂️ Persistent ChromaDB vector database
- 🔤 OCR support for scanned/image-based PDFs
- 🖼️ Image-aware document processing
- ✂️ Configurable document chunking
- 🎯 Top-K relevant chunk retrieval
- 🛡️ Context limiting to prevent excessive LLM input
- 🌐 Simple and interactive Streamlit interface
- 🔐 Environment-variable based API key configuration

>>>>>>> e9caee6b84ce17c1cef660306944a07078fda358

---


<<<<<<< HEAD
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
=======
## 🧠 How It Works
>>>>>>> e9caee6b84ce17c1cef660306944a07078fda358


The application follows a standard RAG pipeline:

<<<<<<< HEAD
| Component | Tool |
|---|---|
| LLM | [Ollama](https://ollama.com) — Qwen2.5 1.5B (runs locally) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | `BAAI/bge-base-en-v1.5` via Hugging Face |
| PDF Parsing | PyMuPDF (`fitz`) |
| Web UI | Streamlit |
| OCR (scanned PDFs) | Tesseract + pytesseract |
=======
>>>>>>> e9caee6b84ce17c1cef660306944a07078fda358

                 ┌─────────────────────┐
                 │     PDF Upload      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   PDF Processing   │
                 │    PyMuPDF/OCR     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Text Extraction  │
                 │   + Image/OCR Data │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Text Chunking     │
                 │  1000 chars        │
                 │  150 overlap       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Embedding Model    │
                 │ all-MiniLM-L6-v2   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     ChromaDB        │
                 │   Vector Storage    │
                 └──────────┬──────────┘
                            │
                       User Question
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Query Embedding     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Semantic Retrieval  │
                 │     Top-K = 8       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Relevant Context    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Groq LLM       │
                 │ Llama 3.3 70B       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Generated Answer  │
                 └─────────────────────┘
🛠️ Tech Stack
Technology	Purpose
Python	Core application logic
Streamlit	Web interface
ChromaDB	Vector database
Sentence Transformers	Text embeddings
Hugging Face	Embedding model
Groq API	Large Language Model inference
PyMuPDF	PDF processing and text extraction
Tesseract OCR	Text extraction from scanned documents
python-dotenv	Environment variable management
📦 Requirements

The project uses the following major dependencies:

<<<<<<< HEAD
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
=======
>>>>>>> e9caee6b84ce17c1cef660306944a07078fda358
streamlit>=1.32.0
chromadb>=1.0.0
sentence-transformers>=3.0.0
ollama>=0.1.0
pymupdf>=1.24.0
python-dotenv>=1.0.0
pytesseract>=0.3.13

Additional dependencies may be installed automatically depending on the versions of the packages above.

📁 Project Structure

<<<<<<< HEAD
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
=======
A typical project structure looks like this:

rag-document-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── chroma_db/
│   └── ...
│
└── assets/
    └── ...

The exact structure may vary depending on the current implementation.

⚙️ Configuration

The application uses environment variables for sensitive configuration such as the Groq API key.

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here

Do not commit the .env file to GitHub.

Add it to .gitignore:

.env
chroma_db/
__pycache__/
*.pyc
🔑 Getting a Groq API Key

The application uses the Groq API for LLM inference.

Create an account and generate an API key from the Groq developer console.

Then add the key to your .env file:

GROQ_API_KEY=your_api_key
🐍 Installation
1. Clone the repository
git clone https://github.com/your-username/rag-document-assistant.git

Move into the project directory:

cd rag-document-assistant
2. Create a virtual environment

Linux/macOS:

python3 -m venv venv

Activate it:

source venv/bin/activate

Windows:

python -m venv venv

Activate:

venv\Scripts\activate
3. Install Python dependencies
pip install -r requirements.txt
🔤 Install Tesseract OCR

OCR is used for scanned PDFs and documents where normal text extraction is insufficient.

Ubuntu/Debian
sudo apt update
sudo apt install tesseract-ocr

Verify the installation:

tesseract --version
Windows

Install Tesseract OCR and make sure the Tesseract executable is available in your system PATH.

Verify:

tesseract --version
▶️ Running the Application

After configuring the environment:

streamlit run app.py

The application will start a local Streamlit server.

Open the displayed address in your browser.

📄 Using the Application
Step 1 — Upload a PDF

Upload a PDF document through the Streamlit interface.

The application processes the document and extracts available content.

Step 2 — Document Processing

The application:

Reads the PDF.
Extracts text using PyMuPDF.
Detects content that may require OCR.
Uses Tesseract for scanned/image-based content.
Splits extracted content into smaller chunks.
Generates embeddings for each chunk.
Stores the embeddings in ChromaDB.
Step 3 — Ask Questions

After processing the document, ask questions in natural language.

For example:

What is this document about?
What are the main objectives mentioned in the document?
Summarize the key findings.
What are the important dates mentioned?
Explain the methodology used in this document.

The system retrieves the most relevant sections of the document before generating the answer.

🧠 RAG Pipeline

The project uses Retrieval-Augmented Generation instead of simply sending the entire document to an LLM.

The process is:

PDF
 │
 ▼
Text / OCR Extraction
 │
 ▼
Chunking
 │
 ▼
Embeddings
 │
 ▼
ChromaDB
 │
 ▼
User Question
 │
 ▼
Question Embedding
 │
 ▼
Similarity Search
 │
 ▼
Top-K Relevant Chunks
 │
 ▼
LLM Context
 │
 ▼
Groq
 │
 ▼
Answer

This approach allows the model to answer questions using information retrieved specifically from the uploaded document.

🔢 Document Chunking

Documents are divided into smaller chunks before generating embeddings.

Current configuration:

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
Chunk Size

Each chunk contains approximately 1000 characters.

Chunk Overlap

Adjacent chunks overlap by approximately 150 characters.

The overlap helps preserve context when important information exists across chunk boundaries.

Example:

Chunk 1:
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


Chunk 2:
              AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

The overlapping section helps prevent important information from being lost during chunking.

🔎 Semantic Search

Instead of searching documents using only exact keywords, the application converts text into numerical vectors called embeddings.

The embedding model used by the project is:

all-MiniLM-L6-v2

This allows semantically similar content to be retrieved even when the wording of the question differs from the wording in the document.

For example:

Question:
"Who is responsible for maintaining the system?"


Document:
"The infrastructure team handles all server maintenance."

Although the exact words differ, the semantic meaning is related.

🗃️ ChromaDB

ChromaDB is used as the vector database.

The project stores document embeddings in a persistent directory:

./chroma_db

The collection used by the application is:

pdf_rag

The stored information can include:

Document chunks
Embeddings
Metadata
Source information

This allows the application to perform similarity searches against previously processed content.

🎯 Retrieval Configuration

The application retrieves the most relevant chunks from the vector database.

Current configuration:

TOP_K = 8

This means the system attempts to retrieve the top 8 relevant chunks for a user query.

The retrieved information is then passed to the LLM as context.

🧠 Context Management

To prevent excessively large prompts from being sent to the LLM, the application limits the amount of retrieved context.

Current configuration:

MAX_CONTEXT_CHARS = 14000

This helps control:

Prompt size
API usage
Response latency
Context relevance
🤖 LLM

The project uses Groq for fast LLM inference.

Configured model:

llama-3.3-70b-versatile

The LLM receives:

System Instructions
        +
User Question
        +
Retrieved Document Context

It then generates an answer based primarily on the retrieved document content.

🔤 OCR Support

Not every PDF contains selectable text.

Some PDFs are essentially collections of scanned images.

For example:

Scanned PDF
     │
     ▼
PDF Image
     │
     ▼
Tesseract OCR
     │
     ▼
Extracted Text
     │
     ▼
RAG Pipeline

Tesseract OCR allows the application to extract text from image-based documents.

This improves compatibility with:

Scanned reports
Digitized books
Scanned forms
Image-based PDFs
Documents created from photographs
🖼️ Image-Aware Processing

The document processing pipeline is designed to account for PDFs containing images and scanned content.

When normal PDF text extraction is insufficient, OCR can be used to extract text from the document images.

This helps the RAG system work with documents where important information is embedded inside scanned pages.

🧪 Example Workflow

Suppose the uploaded document is:

company_policy.pdf

The user asks:

What is the company's leave policy?

The system performs:

1. Convert question into an embedding


2. Search ChromaDB


3. Retrieve relevant document chunks


4. Build context


5. Send question + context to Groq


6. Generate answer


7. Display answer in Streamlit

The LLM does not need to process the entire PDF every time.

Instead, it receives the most relevant sections retrieved from the vector database.

📊 Why RAG?

A conventional chatbot may not know the contents of a private document.

RAG solves this by combining:

Information Retrieval
        +
Large Language Model

Instead of asking the LLM:

"Answer this question."

the application provides:

"Here is the relevant information from the document.
Now answer the question using this context."

This makes the system better suited for document-specific question answering.

🔐 Security Considerations

Sensitive configuration should never be hardcoded.

Avoid:

GROQ_API_KEY = "my-secret-api-key"

Use environment variables instead:

GROQ_API_KEY=your_api_key

And make sure .env is included in .gitignore:

.env

Never commit API keys, passwords, tokens, or other credentials to GitHub.

⚡ Performance Considerations

The application uses several techniques to keep the RAG pipeline efficient:

Semantic vector search
Top-K retrieval
Chunk overlap
Context-size limiting
Persistent vector storage
Fast LLM inference through Groq
Local embedding generation

The goal is to retrieve only the information required to answer the user's question instead of repeatedly processing the entire document.

⚠️ Limitations

Although the application supports a variety of PDF documents, it has some limitations.

Complex PDF layouts

Tables, columns, headers, footers, and unusual layouts may not always be extracted perfectly.

OCR accuracy

OCR quality depends on:

Image resolution
Font
Document quality
Scan quality
Background noise
Images without text

Images containing charts, diagrams, or visual information may require additional vision-based processing to fully understand their content.

Hallucinations

The LLM can still generate incorrect information.

The application therefore relies on retrieval quality and carefully constructed prompts to keep answers grounded in the uploaded document.

Large documents

Very large documents may require additional optimization such as:

Better chunking strategies
Hierarchical retrieval
Metadata filtering
Reranking
Document summarization
🔮 Future Improvements

Possible future improvements include:

🖼️ Vision-language model support for document images
📊 Better table extraction
🔍 Reranking retrieved chunks
📚 Multi-document querying
🗂️ Document management
💾 User-specific document collections
🔐 Authentication
🧹 Automatic duplicate detection
📈 Retrieval evaluation
🧠 Hybrid keyword + semantic search
📝 Citation generation
📄 Page-level source references
💬 Conversation history
⚡ Streaming LLM responses
☁️ Cloud deployment
🐳 Docker containerization
📊 Application monitoring
🐳 Docker Deployment

The application can also be containerized in the future using Docker.

A typical deployment architecture could look like:

                    Internet
                       │
                       ▼
                ┌──────────────┐
                │   Reverse    │
                │    Proxy     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Streamlit   │
                │     App      │
                └──────┬───────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        ┌───────────┐     ┌────────────┐
        │ ChromaDB  │     │  Groq API  │
        └───────────┘     └────────────┘
☁️ Deployment Options

The application can potentially be deployed using:

Streamlit Community Cloud
Docker
AWS EC2
AWS ECS
AWS App Runner
Google Cloud Run
Azure Container Apps
A personal Linux server
Kubernetes

For production deployments, additional security and observability should be implemented.

🧪 Testing

Before deployment, test the application with different types of PDFs:

✓ Normal text PDF
✓ Scanned PDF
✓ Multi-page PDF
✓ PDF with images
✓ PDF with tables
✓ PDF with multiple sections
✓ Large PDF

Also test questions where:

✓ The answer exists in the document
✓ The answer does not exist
✓ Similar wording is used
✓ Different wording is used
✓ Multiple sections contain relevant information
📌 Example Questions

You can test the application with questions such as:

What is the main purpose of this document?
Summarize the document in five points.
What are the key recommendations?
Who are the stakeholders mentioned?
What are the important dates?
What problems are identified in the document?
What solutions are proposed?
Explain this section in simple terms.
📜 Environment Variables
Variable	Description
GROQ_API_KEY	API key used to access Groq LLM inference

Example:

GROQ_API_KEY=xxxxxxxxxxxxxxxx
⚙️ Application Configuration

The main RAG parameters can be adjusted depending on the project requirements.

Current configuration:

CHROMA_PERSIST_DIR = ./chroma_db


COLLECTION_NAME = pdf_rag


EMBEDDING_MODEL = all-MiniLM-L6-v2


CHUNK_SIZE = 1000


CHUNK_OVERLAP = 150


TOP_K = 8


MIN_CHUNK_LEN = 40


MAX_CONTEXT_CHARS = 14000


GROQ_MODEL = llama-3.3-70b-versatile

These parameters can be tuned to improve retrieval quality and performance.

🧩 Troubleshooting
Groq API Key Error

Make sure the .env file exists:

.env

and contains:

GROQ_API_KEY=your_api_key

Restart Streamlit after changing environment variables.

Tesseract Not Found

Check whether Tesseract is installed:

tesseract --version

If it is not installed on Ubuntu:

sudo apt update
sudo apt install tesseract-ocr
ChromaDB Issues

If the vector database becomes corrupted during development, the local database can be removed and recreated:

rm -rf chroma_db

Then restart the application and process the documents again.

Do this only when you are okay with rebuilding the stored document embeddings.

Streamlit Not Found

Make sure the virtual environment is activated:

source venv/bin/activate

Then install the dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py
📈 Project Goals

This project demonstrates practical implementation of:

Retrieval-Augmented Generation
Vector databases
Semantic search
Embedding models
LLM integration
PDF processing
OCR
Prompt construction
Context retrieval
Python application development
Streamlit UI development

It is designed as a practical example of combining traditional document processing with modern generative AI.

👨‍💻 Author

Vishal Anand

Cloud / DevOps Engineer

GitHub:

https://github.com/serverlesswizard

LinkedIn:

https://linkedin.com/in/vishalanand25
⭐ If You Found This Project Useful

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

📄 License

This project is available under the MIT License.

See the LICENSE file for more information.
>>>>>>> e9caee6b84ce17c1cef660306944a07078fda358
