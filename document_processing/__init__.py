"""Document processing pipeline — multi-format loading and chunking."""

from document_processing.loaders import load_document
from document_processing.chunking import chunk_documents

__all__ = ["load_document", "chunk_documents"]
