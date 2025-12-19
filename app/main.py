import os
import traceback
from dotenv import load_dotenv
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


from app.embeddings import *
from app.vector_store import *
from app.rag_pipeline import *
from app.llm import * 

# Load .env from project root (optional)
load_dotenv()

# Initialize env variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INPUT_DIR = os.environ["INPUT_PDF_PATH"]
CHUNKS_DIR = os.environ["CHUNKS_DIR"]
CHUNK_SIZE = os.environ["CHUNK_SIZE"]
CHUNK_OVERLAP = os.environ["CHUNK_OVERLAP"]
FAISS_DIR = os.environ["FAISS_DIR"]


def process(query):
    PROJECT_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
    print(PROJECT_ROOT)
    
    # initialize path for local run
    # full_input_path = Path(str(PROJECT_ROOT) + "\\" + str(INPUT_DIR))
    # print("Input Dir:: ",Path(str(PROJECT_ROOT) + "\\" + str(INPUT_DIR)))

    # initialize path for docker container run
    full_input_path = Path(str(PROJECT_ROOT) + str(INPUT_DIR))
    print("Input Dir:: ",Path(str(PROJECT_ROOT) + str(INPUT_DIR)))
    
    pdfs = find_pdfs(full_input_path, recursive=True)
    
    print("PDF :: ", pdfs)
    print(f"CHUNK_SIZE : {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    
    print("Exists?", full_input_path.exists(), "| Is dir?", full_input_path.is_dir())
    
    if not pdfs:
        
        print(f" No PDFs found under: {full_input_path}")
        sys.exit(0)

    print(f"Found {len(pdfs)} PDF(s). Starting processing...\n")

    for pdf in pdfs:
        print(f"--- Processing: {pdf} ---")
        try:
            pages_info, chunks_info = chunk_pdf(
                pdf_path=pdf,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            save_chunks_hierarchy(
                # output_root=Path(str(PROJECT_ROOT) + "//" + str(CHUNKS_DIR)), # Path for local run
                output_root=Path(str(PROJECT_ROOT) + str(CHUNKS_DIR)), # Path for docker run
                pdf_path=pdf,
                pages_info=pages_info,
                chunks_info=chunks_info,
            )
        except Exception as e:
            print(f"❌ Error processing {pdf}: {e}")
            traceback.print_exc()
        print()

    
    # Read chunk documents
    # docs = read_chunk_documents(Path(str(PROJECT_ROOT) + "//" + str(CHUNKS_DIR))) # For local run
    docs = read_chunk_documents(Path(str(PROJECT_ROOT) + str(CHUNKS_DIR))) # For docker run
    print(f"Loaded {len(docs)} chunk Documents")

    # Build + save FAISS
    # retriever, embeddings = build_and_save_faiss(docs, Path(str(PROJECT_ROOT) + "//" + str(FAISS_DIR))) # For local run
    retriever, embeddings = build_and_save_faiss(docs, Path(str(PROJECT_ROOT) + str(FAISS_DIR))) # For docker run

    # Build Retrieval (RAG) Chain
    rag_chain = rag_pipeline(retriever=retriever)

    # Example query
    # query = "What does the document say about interest rate calculations?"  
    
    response = answer_query_with_openai(
        query=query,
        rag_chain=rag_chain
    )

    print("\n=== ANSWER ===")
    print(response["answer"])

    print("\n=== SOURCES ===")
    for s in response["sources"]:
        print(f"Source : {s}")
    
    return response["sources"], response["answer"]


app = FastAPI()


class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def rag_endpoint(body: Query):
    best_doc, result = process(body.query)

    return {
        "context_used": best_doc,
        "answer": result
    }
