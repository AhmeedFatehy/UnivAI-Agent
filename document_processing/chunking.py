"""Chunking strategies for different document types.

Uses MarkdownTextSplitter for PDF/Markdown (since PyMuPDF4LLM outputs markdown)
and RecursiveCharacterTextSplitter for everything else.
"""
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs: list, file_type: str | None = None) -> list:
    """Split documents into chunks with the best strategy for the file type.

    Args:
        docs: List of LangChain Document objects.
        file_type: File extension without dot (e.g., "pdf", "docx").
                   If None, tries to read from doc metadata.

    Returns:
        List of chunked Document objects with positional metadata.
    """
    if file_type is None and docs:
        file_type = docs[0].metadata.get("document_type", "txt")

    splitter = _get_splitter(file_type)
    chunks = splitter.split_documents(docs)

    # Enrich chunks with positional metadata
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total

    return chunks


def _get_splitter(file_type: str | None):
    """Return the appropriate text splitter for the document type."""
    # PDF and Markdown are converted to markdown by PyMuPDF4LLM,
    # so MarkdownTextSplitter preserves structure better.
    if file_type in ("pdf", "md", "markdown"):
        from langchain_text_splitters import MarkdownTextSplitter
        return MarkdownTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    # For all other formats, use RecursiveCharacterTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
