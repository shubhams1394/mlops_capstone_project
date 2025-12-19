
# MLOPS_PROJECT_BFS – RAG over PDFs with FastAPI & Docker

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline over PDF documents and exposes it via a **FastAPI** service.  
It includes local development instructions, Dockerization, and a GitHub Actions **CI/CD** workflow that builds and publishes the container image when changes are merged to `main`.

---

## Folder Structure

├─ .github/
│  └─ workflows/
│     └─ ci-cd.yaml            # GitHub Actions workflow (CI/CD)
├─ app/
│  ├─ embeddings.py            # Embedding utilities (OpenAI embeddings, etc.)
│  ├─ llm.py                   # LLM client helpers and prompting
│  ├─ main.py                  # FastAPI application (entry module: app.main:app)
│  ├─ rag_pipeline.py          # RAG orchestration: load chunks → retriever → LLM
│  └─ vector_store.py          # Vector store utilities (Chroma/FAISS)
├─ data/
│  ├─ chunks/                  # (Generated) Chunked text hierarchy from PDFs
│  ├─ documents/               # Source PDFs to index
│  └─ faiss_index/             # (Optional) FAISS index persistence
├─ k8s/                        # (Optional) Kubernetes manifests
├─ .env                        # Environment variables (local/dev usage)
├─ Dockerfile                  # Container build file
├─ requirements.txt            # Python dependencies
└─ README.md                   # You are here

## ⚙️ Requirements

- **Python** 3.10+
- (Optional) **Docker** 24+
- (Optional) **GitHub Actions** for CI/CD
- An **OpenAI API key** if you are using OpenAI embeddings/LLM.

---

## 🔐 Environment Variables

Create a `.env` in the project root. Example:

```env
OPENAI_API_KEY=sk-...

INPUT_PDF_PATH=/data/documents
CHUNKS_DIR=/data/chunks
FAISS_DIR=/data/faiss_index
CHUNK_SIZE
CHUNK_OVERLAP

APP_HOST=0.0.0.0
APP_PORT=8000
```

# Local Development

1. Create venv & install deps

```

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run FastAPI
```

uvicorn main:app --port 8000

```

# Run with Docker

## Build
```
docker build -t mlops_capstone_project:latest .
```

## Run
```
docker run -d --name mlops_capstone_project -p 8000:8000 mlops_capstone_project:latest
```

# CI/CD (GitHub Actions)
Workflow at .github/workflows/ci-cd.yaml:

 - On Pull Requests to main: installs dependencies, runs tests, builds Docker image (no push).
 - On Push to main: builds and pushes Docker image to registry (Docker Hub or GHCR).

# RAG Pipeline Overview
1. Ingestion: PDFs → chunks → metadata.
2. Indexing: Embeddings → vector store (Chroma or FAISS).
3. Querying: /ask endpoint retrieves top-k chunks and calls LLM.

# API Endpoints
 - GET /health → Health check
 - POST /ask

 ```
 {
    "query": "What does the document say about interest rate calculations?"
 }
```

Response:

```
{
    "context_used": [
        {
            "source": "/app/data/documents/section3-2.pdf",
            "pdf_stem": null,
            "page_number": 53,
            "chunk_index": null,
            "preview": "becomes $80,000.  If the bank is on an accrual basis of  accounting, there may also be adjusting entries necessary  to reduce both the accrued interest receivable and loan  interest income accounts. A..."
        },
        {
            "source": "/app/data/documents/section3-2.pdf",
            "pdf_stem": null,
            "page_number": 18,
            "chunk_index": null,
            "preview": "LOANS Section 3.2  Loans (8-16) 3.2-18 RMS Manual of Examination Policies    Federal Deposit Insurance Corporation  • Advance rates, risk -adjusted values for PDP, PDNP,  and PUD reserves, and require..."
        },
        {
            "source": "/app/data/documents/section3-2.pdf",
            "pdf_stem": null,
            "page_number": 16,
            "chunk_index": null,
            "preview": "LOANS Section 3.2  Loans (8-16) 3.2-16 RMS Manual of Examination Policies    Federal Deposit Insurance Corporation  As part of the underwriting process, lending personnel  should prepare both base -ca..."
        }
    ],
    "answer": "The document states that if the bank is on an accrual basis of accounting, there may be adjusting entries necessary to reduce both the accrued interest receivable and loan interest income accounts. Additionally, when establishing the \"economic value\" of a new mortgage loan, the value is represented by the sum of the present value of the income stream to be received from the new loan, discounted at the current market rate for this type of credit, and the present value of the principal to be received, also discounted at the current market rate."
}
```