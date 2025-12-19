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

# load_dotenv(dotenv_path=env_path, override=False)
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INPUT_DIR = os.environ["INPUT_PDF_PATH"]
CHUNKS_DIR = os.environ["CHUNKS_DIR"]
CHUNK_SIZE = os.environ["CHUNK_SIZE"]
CHUNK_OVERLAP = os.environ["CHUNK_OVERLAP"]

def find_pdfs(root: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Find all .pdf files under root."""
    root = Path(root)  # normalize to Path
    if recursive:
        return sorted(p for p in root.rglob("*.pdf") if p.is_file())
    else:
        return sorted(p for p in root.glob("*.pdf") if p.is_file())
    
def chunk_pdf(
    pdf_path: Path, chunk_size: int, chunk_overlap: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load a PDF into page-level Documents, split into chunks, and return lists:
    - pages_info: metadata per original page
    - chunks_info: dicts with text + metadata per chunk
    """
    print(f"chunk_size : {chunk_size}, chunk_overlap: {chunk_overlap}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()  # one Document per page

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(CHUNK_SIZE),  # A larger chunk size
        chunk_overlap=int(CHUNK_OVERLAP), # A smaller overlap (e.g., 15% of chunk_size)
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Compute page metadata list (original pages)
    pages_info = []
    for doc in documents:
        md = dict(doc.metadata) if hasattr(doc, "metadata") else {}
        pages_info.append(md)

    # Prepare chunks info with clean metadata
    chunks_info = []
    for i, chunk in enumerate(chunks, start=1):
        md = dict(chunk.metadata) if hasattr(chunk, "metadata") else {}
        # Normalize page numbering to 1-based if present
        page_num = md.get("page")
        if isinstance(page_num, int):
            md["page_number"] = page_num + 1  # make it human-friendly 1-based
        else:
            # Some loaders might store differently; keep original and set default
            md["page_number"] = md.get("page_number") or 1

        chunks_info.append(
            {
                "index": i,
                "text": chunk.page_content,
                "metadata": md,
            }
        )

    print(f"Loaded {len(documents)} pages → {len(chunks)} chunks")
    if chunks_info:
        print("Example chunk metadata:", chunks_info[0]["metadata"])

    return pages_info, chunks_info

def save_chunks_hierarchy(
    output_root: Path,
    pdf_path: Path,
    pages_info: List[Dict[str, Any]],
    chunks_info: List[Dict[str, Any]],
) -> None:
    """
    Save chunks into:
      <output_root>/<pdf_stem>/page-XXXX/chunk-YYYY.txt and chunk-YYYY.metadata.json
    Also write a manifest.json summarizing.
    """
    pdf_stem = safe_name(pdf_path.stem)
    print(f"pdf_stem: {pdf_stem}, output_root: {output_root}")
    base_dir = Path(os.path.join(str(output_root), str(pdf_stem)))
    print("base_dir :: ", base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Group chunks by page_number
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for ch in chunks_info:
        page_num = ch["metadata"].get("page_number", 1)
        by_page.setdefault(int(page_num), []).append(ch)

    total_written = 0
    manifest = {
        "pdf_file": str(pdf_path),
        "pdf_stem": pdf_stem,
        "total_pages": len(pages_info),
        "total_chunks": len(chunks_info),
        "pages": [],
    }

    for page_num in sorted(by_page.keys()):
        page_dir = base_dir / f"page-{page_num:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        page_entry = {
            "page_number": page_num,
            "chunks": [],
        }

        for idx, ch in enumerate(by_page[page_num], start=1):
            # Prepare file paths
            chunk_txt = page_dir / f"chunk-{idx:04d}.txt"
            chunk_meta = page_dir / f"chunk-{idx:04d}.metadata.json"

            # Write text
            chunk_txt.write_text(ch["text"], encoding="utf-8")

            # Write metadata (include text length, index for convenience)
            meta_out = {
                "index": ch["index"],
                "page_number": page_num,
                "text_length": len(ch["text"]),
                "metadata": ch["metadata"],
            }
            chunk_meta.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

            page_entry["chunks"].append(
                {
                    "text_file": str(chunk_txt),
                    "metadata_file": str(chunk_meta),
                }
            )
            total_written += 1

        manifest["pages"].append(page_entry)

    # Save manifest.json at the PDF root
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Saved {total_written} chunk files under: {base_dir}")

def safe_name(name: str, max_len: int = 80) -> str:
    """
    Sanitize a filename component: keep alphanumerics, dot, underscore, hyphen.
    Replace others with '_', trim length.
    """
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if not name:
        name = "untitled"
    return name[:max_len]

def read_chunk_documents(root: Path) -> List[Document]:
    """
    Walk the chunk hierarchy and turn each chunk .txt into a LangChain Document.
    If a .metadata.json exists alongside, merge that into Document.metadata.
    """
    docs: List[Document] = []
    for pdf_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        # pdf_dir = <root>/<PDF_STEM>
        for page_dir in sorted(d for d in pdf_dir.iterdir() if d.is_dir() and d.name.startswith("page-")):
            for txt_path in sorted(page_dir.glob("chunk-*.txt")):
                # Try to find matching metadata json
                meta_path = txt_path.with_name(txt_path.stem + ".metadata.json")

                # Read text
                text = txt_path.read_text(encoding="utf-8")

                # Base metadata
                metadata = {
                    "source_pdf": pdf_dir.name,
                    "page_dir": page_dir.name,
                    "chunk_file": str(txt_path),
                }

                # Merge metadata json if present
                if meta_path.exists():
                    try:
                        meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
                        # Flatten useful bits
                        metadata.update({
                            "page_number": meta_json.get("page_number"),
                            "index": meta_json.get("index"),
                            "text_length": meta_json.get("text_length"),
                        })
                        # Include original metadata dict if available
                        if "metadata" in meta_json and isinstance(meta_json["metadata"], dict):
                            metadata.update(meta_json["metadata"])
                    except Exception:
                        # If metadata parse fails, continue with base metadata
                        pass

                docs.append(Document(page_content=text, metadata=metadata))
    return docs

