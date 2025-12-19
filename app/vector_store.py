import os
import argparse
import json
import os
import re
import traceback
from dotenv import load_dotenv
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def build_and_save_faiss(docs: List[Document], faiss_dir: Path):
    """
    Create embeddings using OpenAI and build a FAISS index, then save locally.
    """
    # Ensure API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in .env or OS env vars.")

    # Create embeddings (small, fast, inexpensive)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("FAISS index ready with", vectorstore.index.ntotal, "vectors")

    # Save locally
    Path(faiss_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(faiss_dir))
    print(f"Saved FAISS index to: {faiss_dir}")

    # Return retriever for immediate use
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever, embeddings


def load_faiss(faiss_dir: Path, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    """
    vs = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS index with", vs.index.ntotal, "vectors")
    return vs