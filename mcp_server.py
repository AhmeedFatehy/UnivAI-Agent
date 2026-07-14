"""MCP Server for the RAG Module.

Exposes RAG capabilities as tools for agents over HTTP.
"""
import os
import json
from mcp.server.fastmcp import FastMCP

from retrieval.pipeline import retrieve_formatted
from document_processing.loaders import load_document
from document_processing.chunking import chunk_documents
from vector_store.indexing import index_chunks
from vector_store.collection_manager import list_user_documents, delete_document
from evaluation.metrics import evaluate_retrieval

# Create the MCP server instance
mcp = FastMCP("rag-module")


@mcp.tool()
def retrieve_context(
    query: str, 
    user_id: str, 
    limit: int = 5,
    use_reranking: bool = True,
    use_query_transform: bool = False
) -> str:
    """Retrieve relevant documents from the user's knowledge base.
    
    Uses hybrid search (dense + sparse), RRF fusion, and optional reranking/transform.

    Args:
        query: The natural-language search query.
        user_id: The ID of the user/student making the query.
        limit: Maximum number of chunks to return (default 5).
        use_reranking: Whether to use cross-encoder reranking (better quality).
        use_query_transform: Whether to decompose complex queries into sub-queries.
    """
    try:
        return retrieve_formatted(
            query=query,
            user_id=user_id,
            limit=limit,
            use_reranking=use_reranking,
            use_query_transform=use_query_transform
        )
    except Exception as e:
        return f"Error during retrieval: {str(e)}"


@mcp.tool()
def ingest_file(file_path: str, user_id: str) -> str:
    """Ingest a document file (PDF, DOCX, TXT, HTML, MD) into the user's knowledge base.
    
    Args:
        file_path: Absolute path to the file to ingest.
        user_id: The ID of the user/student uploading the file.
    """
    try:
        # 1. Load
        docs = load_document(file_path)
        if not docs:
            return "Failed to load document or document is empty."
            
        doc_type = docs[0].metadata.get("document_type", "unknown")
        filename = os.path.basename(file_path)
        
        # 2. Chunk
        chunks = chunk_documents(docs, doc_type)
        
        # 3. Index
        result = index_chunks(
            chunks=chunks,
            user_id=user_id,
            source_filename=filename,
            document_type=doc_type
        )
        
        return f"Successfully ingested {filename}. Created {result['chunks_indexed']} chunks. Document ID: {result['document_id']}"
    except Exception as e:
        return f"Error during ingestion: {str(e)}"


@mcp.tool()
def list_documents(user_id: str) -> str:
    """List all documents currently available in the user's knowledge base.
    
    Args:
        user_id: The ID of the user/student.
    """
    try:
        docs = list_user_documents(user_id)
        if not docs:
            return f"No documents found for user '{user_id}'."
            
        return json.dumps(docs, indent=2)
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@mcp.tool()
def remove_document(user_id: str, document_id: str) -> str:
    """Delete a document and all its chunks from the knowledge base.
    
    Args:
        user_id: The ID of the user/student.
        document_id: The UUID of the document to delete (get this from list_documents).
    """
    try:
        deleted = delete_document(user_id, document_id)
        return f"Successfully deleted document '{document_id}'. Removed {deleted} chunks."
    except Exception as e:
        return f"Error deleting document: {str(e)}"


@mcp.tool()
def server_info() -> str:
    """Return basic info about the RAG MCP server and available tools."""
    return (
        "RAG MCP server is running.\n"
        "Available Tools:\n"
        "- retrieve_context: Search the knowledge base\n"
        "- ingest_file: Add a document to the knowledge base\n"
        "- list_documents: See all uploaded documents\n"
        "- remove_document: Delete a document\n"
        "Supports multi-tenant isolation via user_id metadata filtering."
    )


def preload_models():
    """Pre-load embedding and reranking models at startup."""
    import logging
    from vector_store.qdrant_client import get_dense_embedder, get_sparse_embedder
    from retrieval.reranker import _get_reranker
    from retrieval.query_transform import _get_llm
    
    logger = logging.getLogger(__name__)
    logger.info("Pre-loading models...")
    
    # Pre-load embeddings
    get_dense_embedder()
    get_sparse_embedder()
    
    # Pre-load reranker
    _get_reranker()
    
    # Pre-load LLM (for query transform)
    _get_llm()
    
    logger.info("All models pre-loaded successfully.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Preload models so the first request doesn't timeout
    preload_models()
    
    # Run the server over HTTP using streamable-http transport
    mcp.run(transport="streamable-http")