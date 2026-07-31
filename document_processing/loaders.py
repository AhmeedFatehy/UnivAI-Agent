"""Multi-format document loading.

Supports: PDF, DOCX, TXT, HTML, Markdown.
Auto-detects format from file extension and enriches metadata.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from config import SUPPORTED_FORMATS


def load_document(file_path: str) -> list:
    """Load a document from file path. Auto-detects format by extension.

    Args:
        file_path: Path to the document file.

    Returns:
        List of LangChain Document objects with enriched metadata.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: '{ext}'. Supported: {SUPPORTED_FORMATS}"
        )

    loader = _get_loader(ext, file_path)
    docs = loader.load()

    # Enrich every document with common metadata
    for doc in docs:
        doc.metadata.update({
            "source_filename": path.name,
            "document_type": ext.lstrip("."),
            "file_size_bytes": os.path.getsize(file_path),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        })

    return docs


def _get_loader(ext: str, file_path: str):
    """Return the appropriate LangChain loader for the given extension."""
    if ext == ".pdf":
        from langchain_pymupdf4llm import PyMuPDF4LLMLoader
        return PyMuPDF4LLMLoader(file_path=file_path)

    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path=file_path)

    if ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path=file_path)

    if ext in (".html", ".htm"):
        from langchain_community.document_loaders import BSHTMLLoader
        return BSHTMLLoader(file_path=file_path)

    if ext in (".md", ".markdown"):
        # TextLoader, not UnstructuredMarkdownLoader: Unstructured renders the
        # Markdown to plain text and drops the '#' headings with it, which costs
        # every chunk its section and every book its title (both fall back to the
        # filename). Chunking already uses MarkdownTextSplitter for this type and
        # document_processing.metadata reads the headings, so the raw source is
        # what the rest of the pipeline actually wants.
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path=file_path, autodetect_encoding=True)

    raise ValueError(f"No loader configured for extension: {ext}")
