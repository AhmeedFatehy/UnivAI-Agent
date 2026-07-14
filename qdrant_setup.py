from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid
from load_documents import load_and_split_pdf

def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.
    
    Uses FastEmbed for generating embeddings locally.
    """
    # TODO: Initialize Native Qdrant Client
    client = QdrantClient(url="http://localhost:6333")

    # TODO: Initialize embedding models (FastEmbed)
    dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"  # e.g., "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25" # e.g., "Qdrant/bm25" or "prithivida/Splade_PP_en_v1"
    
    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    # TODO: Create collection with explicit vector configurations
    collection_name = "hybrid_document_collection"
    
    # Get embedding dimensions by encoding a sample text
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)
    
    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
    
    # TODO: Prepare points for upload with dense + sparse vectors
    points = []
    texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings in batches
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))
    
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        # Convert sparse vector to Qdrant format
        # FastEmbed returns sparse vectors with .indices and .values attributes
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values = sparse_vec.values.tolist()
        
        point = models.PointStruct(
            id=str(uuid.uuid4()),  # or use idx for integer IDs
            vector={
                "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            },
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
                "chunk_id": idx
            }
        )
        points.append(point)
    
    # TODO: Upload points using native upload_points (with parallelization)
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,      # e.g., 64 (default)
        parallel=1,        # e.g., 2 workers
        max_retries=2,     # e.g., 3
        wait=False           # Async mode for better performance
    )
    
    return client, collection_name, dense_embedder, sparse_embedder

file_path = 'documents/RHSA1.pdf'
chunks = load_and_split_pdf(file_path=file_path)
setup_qdrant_collection(chunks=chunks)