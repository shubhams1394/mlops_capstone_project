from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import openai
import os
from dotenv import load_dotenv
from langchain.schema import Document

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Load .env from project root (optional)
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# --- OpenAI setup ---
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI()


def answer_query_with_openai(
    query: str,
    rag_chain
) -> Dict[str, Any]:
    """
    High-level helper that:
      1) reads chunk documents from disk (if building),
      2) builds or loads a Chroma index,
      3) creates a RAG chain,
      4) returns the LLM answer + source snippets/paths.

    Returns:
      {
        "answer": str,
        "sources": List[Dict[str, Any]],  # metadata for retrieved docs
      }
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env at project root.")

    # Invoke chain
    result = rag_chain.invoke({"input": query})

    # The result typically contains: {"answer": "...", "context": [Document, ...]}
    answer = result.get("answer", "")
    context_docs: List[Document] = result.get("context", [])

    # Collect minimal source info
    sources = []
    for d in context_docs:
        md = dict(d.metadata or {})
        sources.append(
            {
                "source": md.get("source"),
                "pdf_stem": md.get("pdf_stem"),
                "page_number": md.get("page_number"),
                "chunk_index": md.get("chunk_index"),
                "preview": d.page_content[:200].replace("\n", " ") + ("..." if len(d.page_content) > 200 else ""),
            }
        )

    return {"answer": answer, "sources": sources}

