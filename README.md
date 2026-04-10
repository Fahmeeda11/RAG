# Ask Your PDFs — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about their content. The app finds the most relevant section from your documents using vector similarity search and uses GPT to generate a grounded, accurate answer with source attribution.

Built with **LangChain**, **ChromaDB**, **OpenAI**, and **Streamlit**.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [RAG Pipeline Explained](#rag-pipeline-explained)
- [Configuration Defaults](#configuration-defaults)
- [Key Technologies](#key-technologies)

---

## How It Works

```
PDFs ──> Extract Text ──> Chunk ──> Embed ──> Store in ChromaDB
                                                      |
User Question ──> Embed ──> Similarity Search ────────┘
                                                      |
                                         Context + Question ──> GPT ──> Answer + Source
```

1. **Ingest** — PDFs are loaded, their text is extracted page by page, split into small overlapping chunks, converted into numerical embeddings, and stored in a ChromaDB vector database.
2. **Retrieve** — When you ask a question, it gets embedded using the same model. ChromaDB finds the most semantically similar chunk from all stored documents.
3. **Generate** — The retrieved chunk is injected into a prompt template and sent to GPT-3.5-turbo, which generates an answer based strictly on that context.
4. **Display** — The answer and its source document are shown in a Streamlit chat interface.

---

## Architecture

The project follows a three-file pipeline:

```
┌──────────┐        ┌──────────┐        ┌──────────────┐
│  rag.py  │──────> │ ChromaDB │ <──────│   query.py   │
│ (Ingest) │        │ (Vector  │        │ (Retrieve +  │
│          │        │  Store)  │        │  Generate)   │
└──────────┘        └──────────┘        └──────┬───────┘
                                               │
                                        ┌──────┴───────┐
                                        │ streamlit.py │
                                        │    (Chat UI) │
                                        └──────────────┘
```

### `rag.py` — Ingestion Script

- Recursively loads all PDFs from a configured data directory
- Splits documents into 800-character chunks with 80-character overlap using `RecursiveCharacterTextSplitter`
- Embeds chunks using OpenAI's `text-embedding-3-large` model
- Stores embeddings in ChromaDB in batches of 1000
- Run this once to populate the vector store, or re-run when you add new PDFs

### `query.py` — Query Module

- Loads the ChromaDB collection at import time
- Exposes `get_response(query)` — the core function used by the UI
- Performs similarity search (k=1) to find the most relevant chunk
- Formats the result into a prompt template that instructs GPT to answer only from the given context
- Calls `ChatOpenAI` (GPT-3.5-turbo) and returns the response along with source metadata

### `streamlit.py` — Web UI

- Imports `get_response` from `query.py`
- Provides a file upload widget for PDFs (uploaded files are chunked and embedded on the fly)
- Chat interface with full message history stored in `session_state`
- Source attribution displayed in expandable sections below each answer

---

## Project Structure

```
RAG/
├── rag.py                  # Batch PDF ingestion script
├── query.py                # Query engine (similarity search + GPT generation)
├── streamlit.py            # Streamlit chat UI with PDF upload
├── rough.py                # Scratch/experimental file (not part of main app)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, paths)
├── .streamlit/
│   └── secrets.toml        # Streamlit-specific secrets (OpenAI key)
├── chroma/                 # Persisted ChromaDB vector store (auto-generated)
├── doc_files/              # Directory where uploaded PDFs are saved
└── venv/                   # Python virtual environment
```

---

## Prerequisites

- **Python 3.9+**
- **OpenAI API key** — Get one from [OpenAI Platform](https://platform.openai.com/api-keys). You need access to:
  - `text-embedding-3-large` (embedding model)
  - `gpt-3.5-turbo` (chat model)
- **PDF files** you want to query

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows (Git Bash)
source venv/Scripts/activate

# Windows (CMD)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

### `.env` (project root)

Create a `.env` file with the following variables:

```env
USER_AGENT=your-app-name
DATA_PATH=./path/to/your/pdf/folder
OPENAI_API_KEY=your-openai-api-key
CHROMA_PATH=./chroma
```

| Variable | Description |
|----------|-------------|
| `USER_AGENT` | Identifier string for HTTP requests |
| `DATA_PATH` | Path to the folder containing PDFs for batch ingestion via `rag.py` |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `CHROMA_PATH` | Path where ChromaDB persists the vector store |

### `.streamlit/secrets.toml`

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

This is used by Streamlit to authenticate with OpenAI when running the chat UI.

---

## Usage

### Option 1: Batch Ingest + Chat

Use this when you have a folder full of PDFs you want to ingest all at once.

```bash
# Step 1: Ingest all PDFs from DATA_PATH into ChromaDB
python rag.py

# Step 2: Launch the chatbot UI
streamlit run streamlit.py
```

`rag.py` will prompt you for your API key, then process all PDFs in the configured directory.

### Option 2: Upload Through the UI

Use this for a more interactive workflow.

```bash
streamlit run streamlit.py
```

1. Open the URL shown in the terminal (usually `http://localhost:8501`)
2. Upload one or more PDFs using the file upload widget
3. The app automatically chunks, embeds, and stores the uploaded documents
4. Start asking questions in the chat input
5. Each answer includes an expandable **Sources** section showing where the information came from

---

## RAG Pipeline Explained

For those learning about RAG, here is what happens at each stage:

### 1. Document Loading
`PyPDFLoader` reads each PDF and extracts text page by page. It also captures metadata like the source filename and page number, which is used later for source attribution.

### 2. Text Chunking
Raw text is split into smaller pieces using `RecursiveCharacterTextSplitter`. It tries to break at natural boundaries (paragraphs, then sentences, then words) while keeping each chunk under 800 characters. The 80-character overlap between consecutive chunks ensures important context isn't lost at boundaries.

### 3. Embedding
Each chunk is converted into a high-dimensional vector using OpenAI's `text-embedding-3-large` model. These vectors capture semantic meaning — text chunks with similar meaning produce similar vectors, regardless of exact wording.

### 4. Vector Storage
Embeddings are stored in ChromaDB, a vector database optimized for similarity search. Data is persisted to disk in the `chroma/` directory so it survives between restarts.

### 5. Similarity Search
When you ask a question, your question is embedded using the same model. ChromaDB compares your question's vector against all stored vectors and returns the closest match (k=1).

### 6. Prompt Construction
The retrieved chunk is inserted into a prompt template:
```
Answer the question based only on the following context:
{retrieved chunk}
---
Answer the question based on the above context: {your question}
```
This grounding step prevents hallucination and keeps answers faithful to your documents.

### 7. Response Generation
The prompt is sent to GPT-3.5-turbo, which generates a natural language answer. The source document name is returned alongside the answer for verification.

---

## Configuration Defaults

| Setting | Value | Location |
|---------|-------|----------|
| Chunk size | 800 characters | `rag.py`, `streamlit.py` |
| Chunk overlap | 80 characters | `rag.py`, `streamlit.py` |
| Embedding model | `text-embedding-3-large` | `rag.py`, `query.py` |
| Chat model | `gpt-3.5-turbo` | `query.py` |
| Similarity search results (k) | 1 | `query.py` |
| Ingestion batch size | 1000 | `rag.py` |

---

## Key Technologies

| Technology | Purpose |
|------------|---------|
| [LangChain](https://python.langchain.com/) | Framework for chaining LLM operations — text splitting, prompt templates, model calls |
| [ChromaDB](https://www.trychroma.com/) | Open-source vector database for storing and searching document embeddings |
| [OpenAI API](https://platform.openai.com/) | Embedding model (`text-embedding-3-large`) and chat model (`gpt-3.5-turbo`) |
| [Streamlit](https://streamlit.io/) | Python framework for building interactive web apps |
| [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdf/) | LangChain integration for extracting text from PDF files |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Loads environment variables from `.env` files |

---

## License

This project is for educational and learning purposes.
