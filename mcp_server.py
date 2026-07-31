"""MCP Server for the RAG Module.

Exposes RAG capabilities as tools for agents over HTTP.
"""
import os
import json
from mcp.server.fastmcp import FastMCP

from runtime import RuntimeMode, runtime_mode
from retrieval.pipeline import retrieve_formatted
from document_processing.loaders import load_document
from document_processing.chunking import chunk_documents
from vector_store.indexing import index_chunks
from vector_store.collection_manager import list_user_documents, delete_document
from evaluation.metrics import evaluate_retrieval
from tools.registry import TOOL_REGISTRY, TOOL_SCHEMA_VERSION, call_tool

# Create the MCP server instance. The defaults preserve the integrated contract;
# environment overrides let bounded standalone smoke tests avoid occupied ports.
mcp = FastMCP(
    "rag-module",
    host=os.environ.get("FASTMCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("FASTMCP_PORT", "8000")),
)


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
        if runtime_mode() is RuntimeMode.STANDALONE:
            from standalone_store import retrieve

            return retrieve(query, user_id, limit)
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
        if runtime_mode() is RuntimeMode.STANDALONE:
            from standalone_store import ingest

            result = ingest(file_path, user_id)
            filename = os.path.basename(file_path)
            return (
                f"Successfully ingested {filename}. Created "
                f"{result['chunks_indexed']} chunks. Document ID: "
                f"{result['document_id']}"
            )
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
        if runtime_mode() is RuntimeMode.STANDALONE:
            from standalone_store import list_documents as standalone_list

            docs = standalone_list(user_id)
            if not docs:
                return f"No documents found for user '{user_id}'."
            return json.dumps(docs, indent=2)
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
        if runtime_mode() is RuntimeMode.STANDALONE:
            from standalone_store import remove

            deleted = remove(user_id, document_id)
            return f"Successfully deleted document '{document_id}'. Removed {deleted} chunks."
        deleted = delete_document(user_id, document_id)
        return f"Successfully deleted document '{document_id}'. Removed {deleted} chunks."
    except Exception as e:
        return f"Error deleting document: {str(e)}"


@mcp.tool()
def ingest_collection(file_paths: list[str], collection_id: str, user_id: str) -> str:
    """Ingest several books into one collection, tolerating per-book failure.

    Each book is loaded, chunked and indexed independently with its own
    document identity, so one unreadable file does not cost the others their
    index. Transient failures are retried; permanent ones are reported.

    Args:
        file_paths: Absolute paths of the documents to ingest.
        collection_id: Logical collection (e.g. a programme) the books belong to.
        user_id: The ID of the user/student who owns the collection.
    """
    try:
        report = call_tool(
            "ingest_collection",
            {
                "paths": file_paths,
                "collection_id": collection_id,
                "user_id": user_id,
            },
        )
        return report.model_dump_json(indent=2)
    except Exception as e:
        return f"Error during collection ingestion: {str(e)}"


@mcp.tool()
def retrieve_grounded_context(
    query: str,
    user_id: str,
    collection_id: str | None = None,
    document_ids: list[str] | None = None,
    limit: int = 5,
) -> str:
    """Retrieve passages with book/page/section citations, or refuse explicitly.

    Unlike retrieve_context, this returns structured JSON: either cited passages
    or a refusal stating that the indexed material does not cover the question.
    Use it when the answer must be attributable.

    Args:
        query: The natural-language question.
        user_id: The ID of the user/student making the query.
        collection_id: Restrict to one logical multi-book collection.
        document_ids: Restrict to specific documents.
        limit: Maximum number of passages to return.
    """
    try:
        context = call_tool(
            "retrieve_context",
            {
                "query": query,
                "user_id": user_id,
                "collection_id": collection_id,
                "document_ids": document_ids or [],
                "limit": limit,
            },
        )
        return context.model_dump_json(indent=2)
    except Exception as e:
        return f"Error during grounded retrieval: {str(e)}"


@mcp.tool()
def create_programme_plan(
    programme_title: str,
    collection_id: str,
    user_id: str,
    seed_queries: list[str],
    capacity_hours: float = 120.0,
    max_semesters: int = 8,
) -> str:
    """Plan a programme from an indexed collection through the agent graph.

    Runs Manager → Curriculum → Content → Assessment and returns the plan, the
    cited lectures and questions, every refusal and the task trace. Requires a
    reachable LLM (LLM_MODEL at LLM_BASE_URL).

    Args:
        programme_title: Name of the programme to plan.
        collection_id: The indexed collection to plan from.
        user_id: The ID of the user/student who owns the collection.
        seed_queries: Subject areas to retrieve evidence for.
        capacity_hours: Hours one semester can absorb.
        max_semesters: Upper bound on semesters.
    """
    try:
        from agents.graph import run_programme
        from agents.manager import AgentRuntime, ProgrammeRequest, ollama_llm

        result = run_programme(
            ProgrammeRequest(
                programme_title=programme_title,
                collection_id=collection_id,
                user_id=user_id,
                seed_queries=seed_queries,
                capacity_hours=capacity_hours,
                max_semesters=max_semesters,
            ),
            AgentRuntime(llm=ollama_llm()),
        )
        return result.model_dump_json(indent=2)
    except Exception as e:
        return f"Error during programme planning: {str(e)}"


@mcp.tool()
def get_source_location(
    user_id: str, document_id: str, chunk_index: int | None = None
) -> str:
    """Resolve a citation back to its book, page, section and indexed excerpt.

    Args:
        user_id: The ID of the user/student who owns the document.
        document_id: The document the citation points at.
        chunk_index: The specific chunk, if the citation names one.
    """
    try:
        located = call_tool(
            "get_source_location",
            {
                "user_id": user_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
            },
        )
        return located.model_dump_json(indent=2)
    except Exception as e:
        return f"Error resolving source location: {str(e)}"


@mcp.tool()
def server_info() -> str:
    """Return basic info about the RAG MCP server and available tools."""
    return (
        "RAG MCP server is running.\n"
        f"Mode: {runtime_mode().value}\n"
        "Available Tools:\n"
        "- retrieve_context: Search the knowledge base\n"
        "- ingest_file: Add a document to the knowledge base\n"
        "- list_documents: See all uploaded documents\n"
        "- remove_document: Delete a document\n"
        "- ingest_collection: Index several books under one collection identity\n"
        "- retrieve_grounded_context: Cited passages, or an explicit refusal\n"
        "- create_programme_plan: Plan a programme through the agent graph\n"
        "- get_source_location: Resolve a citation back to book/page/section\n"
        "Supports multi-tenant isolation via user_id metadata filtering.\n"
        f"Typed tool contracts: {', '.join(sorted(TOOL_REGISTRY))} "
        f"(schema v{TOOL_SCHEMA_VERSION})."
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
    
    mode = runtime_mode()
    logging.info("Starting Agent MCP in %s mode", mode.value)
    if mode is RuntimeMode.INTEGRATED:
        # Preload models so the first request doesn't timeout.
        preload_models()
    else:
        logging.info("Standalone mode: model preload and Qdrant are disabled")
    
    # Run the server over HTTP using streamable-http transport
    mcp.run(transport="streamable-http")
