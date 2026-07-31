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


def ingest_many(file_paths: list[str], user_id: str, collection_id: str) -> int:
    """Ingest several books into one collection, reporting per-book outcomes.

    A failure on one book does not stop the others. Returns a process exit code:
    0 when every book was indexed, 1 when any failed.
    """
    from document_processing.batch_ingestion import ingest_collection

    report, _ = ingest_collection(
        file_paths, collection_id=collection_id, user_id=user_id
    )

    print(f"\nCollection: {collection_id}")
    print(f"User ID: {user_id}")
    for result in report.succeeded:
        print(
            f"  ✅ {result.book_title} — {result.chunks_indexed} chunks, "
            f"{result.pages} page(s), document {result.document_id} "
            f"(attempt {result.attempts})"
        )
    for failure in report.failed:
        kind = "transient" if failure.retryable else "permanent"
        print(
            f"  ❌ {failure.path} — {failure.error_type}: {failure.error} "
            f"({kind}, {failure.attempts} attempt(s))"
        )

    print(f"\n{report.summary()}")
    return 0 if report.complete_success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG knowledge base.")
    parser.add_argument(
        "file_path",
        nargs="+",
        help="Path(s) to the document(s) (PDF, DOCX, TXT, HTML, MD)",
    )
    parser.add_argument("--user", required=True, help="User ID to associate the document with")
    parser.add_argument(
        "--collection",
        help=(
            "Collection ID to ingest into. Required for more than one file; "
            "gives every chunk a collection identity so retrieval can be scoped "
            "to this set of books."
        ),
    )

    args = parser.parse_args()

    if args.collection:
        # Batch mode reports missing files per book instead of aborting.
        sys.exit(ingest_many(args.file_path, args.user, args.collection))

    if len(args.file_path) > 1:
        print("Error: --collection is required when ingesting more than one file.")
        sys.exit(1)

    if not os.path.exists(args.file_path[0]):
        print(f"Error: File not found: {args.file_path[0]}")
        sys.exit(1)

    ingest(args.file_path[0], args.user)
