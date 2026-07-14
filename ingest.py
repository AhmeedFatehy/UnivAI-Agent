"""CLI script to manually ingest documents into the knowledge base."""
import sys
import argparse
import os

from document_processing.loaders import load_document
from document_processing.chunking import chunk_documents
from vector_store.indexing import index_chunks


def ingest(file_path: str, user_id: str):
    """Ingest a file into Qdrant for a specific user."""
    print(f"Loading document: {file_path}")
    docs = load_document(file_path)
    if not docs:
        print("Error: Document is empty or could not be loaded.")
        return

    doc_type = docs[0].metadata.get("document_type", "unknown")
    filename = os.path.basename(file_path)
    
    print(f"Loaded {len(docs)} pages/elements. Chunking (type: {doc_type})...")
    chunks = chunk_documents(docs, doc_type)
    print(f"Created {len(chunks)} chunks. Indexing into Qdrant...")

    result = index_chunks(
        chunks=chunks,
        user_id=user_id,
        source_filename=filename,
        document_type=doc_type
    )

    print("\n✅ Ingestion Complete!")
    print(f"File: {filename}")
    print(f"User ID: {user_id}")
    print(f"Document ID: {result['document_id']}")
    print(f"Chunks Indexed: {result['chunks_indexed']}")
    print(f"Collection: {result['collection_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG knowledge base.")
    parser.add_argument("file_path", help="Path to the document (PDF, DOCX, TXT, HTML, MD)")
    parser.add_argument("--user", required=True, help="User ID to associate the document with")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
        
    ingest(args.file_path, args.user)