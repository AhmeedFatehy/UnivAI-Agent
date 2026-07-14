# 📚 Lab: Building an Agentic RAG System

Welcome to the **Agentic Retrieval-Augmented Generation (RAG) Lab**! In this lab, you will build a local, privacy-first AI application that uses an **intelligent agent** to decide when to retrieve documents and answer questions about them.

Unlike simple RAG chains, an **Agentic RAG** system uses a ReAct-style agent that can:
- 🤔 **Decide** whether it needs to search for documents or can answer directly
- 🔍 **Retrieve** context only when needed
- 🧠 **Reason** through complex queries requiring multiple searches
- 💬 **Stream** responses in real-time with the latest streaming API

We will be using a powerful modern AI stack:

- **Ollama**: To run the qwen3:4b-instruct Large Language Model locally.
- **Docker & Qdrant**: To host our Vector Database natively for fast, scalable document retrieval.
- **FastEmbed**: To generate highly accurate embeddings using Jina embeddings.
- **LangChain Agentic RAG**: Using `create_agent` (2026 syntax) with retrieval tools.
- **LangGraph**: For agent orchestration with streaming support.
- **uv**: For lightning-fast Python project and dependency management.
- **Streamlit**: To build a beautiful, interactive web user interface.

**Important**: We will use **native Qdrant client** (NOT langchain-qdrant) for:
- Collection creation with explicit vector configurations
- Points insertion using `upload_points`
- Hybrid search with RRF fusion using `prefetch` and `query_points`

Follow the tasks below step-by-step to complete your Agentic RAG system.

---

## 🛠️ Task 1: Prerequisites & Repository Setup

Before we start coding, let's set up our workspace.

1. Ensure you have Git installed on your machine.
2. Clone this repository to your local machine:
   ```bash
   git clone <YOUR_REPO_URL_HERE>
   cd <REPO_NAME>
   ```
3. Create a folder named `documents` to temporarily store your test PDF files.

---

## 🐳 Task 2: Install & Verify Docker

We need Docker to run our Qdrant vector database in an isolated container.

🔗 Reference: [Docker Official Installation Guide](https://docs.docker.com/get-docker/)

1. Download and install Docker Desktop for your operating system.
2. Open your terminal and verify the installation:
   ```bash
   docker --version
   ```

---

## 🗄️ Task 3: Set up Qdrant Vector Database

Qdrant is a high-performance vector search engine. We will run it locally using Docker.

🔗 Reference: [Qdrant Quickstart Documentation](https://qdrant.tech/documentation/quickstart/)

1. Open your terminal and pull the Qdrant Docker image:
   ```bash
   docker pull qdrant/qdrant
   ```

2. Run the Qdrant container, exposing the necessary ports:

   **Mac/Linux (bash):**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage:z \
       qdrant/qdrant
   ```

   **Windows (PowerShell):**
   ```powershell
   docker run -p 6333:6333 -p 6334:6334 `
       -v "${PWD}/qdrant_storage:/qdrant/storage:z" `
       qdrant/qdrant
   ```

   **Windows (cmd):**
   ```cmd
   docker run -p 6333:6333 -p 6334:6334 -v "%cd%/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
   ```

3. Verify it is running at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

---

## 🦙 Task 4: Install & Configure Ollama

Ollama allows us to run large language models locally.

🔗 Reference: [Ollama Download Page](https://ollama.com/download)

1. Download and install Ollama for your OS.
2. Verify the installation:
   ```bash
   ollama --version
   ```

---

## 🧠 Task 5: Download the Qwen3 4B Instruct Model

1. Download the model:
   ```bash
   ollama run qwen3:4b-instruct
   ```
2. Type "Hello" to test, then `/bye` to exit.

---

## 🐍 Task 6: Python Environment & Dependencies

We will use **uv** for Python project management.

🔗 Reference: [uv Official Documentation](https://docs.astral.sh/uv/)

1. **Install uv**:
   - **Mac/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

2. **Initialize the project**:
   ```bash
   uv init
   ```

3. **Rename** `hello.py` to `app.py`.

4. **Add dependencies** (2026 versions):
   ```bash
   uv add streamlit "qdrant-client[fastembed]" fastembed pymupdf4llm langchain-pymupdf4llm langchain-text-splitters langchain-ollama langchain "langchain-core>=1.0" langgraph
   ```
   
   > **Note**: We use **native qdrant-client** (NOT langchain-qdrant) for collection management and hybrid search!

---

## 📄 Task 7: Document Loading & Text Splitting

**Your Task:** Fill in the blanks to create the document loading function:

```python
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

def load_and_split_pdf(file_path):
    # TODO: Initialize PyMuPDF4LLMLoader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=___)
    
    # TODO: Load the document
    docs = loader.___()
    
    # TODO: Initialize MarkdownTextSplitter with appropriate chunk settings
    splitter = MarkdownTextSplitter(chunk_size=___, chunk_overlap=___)
    
    # TODO: Split the documents into chunks
    chunks = splitter.split_documents(___)
    
    return chunks
```

---

## 🧩 Task 8: Native Qdrant Setup - Collection Creation & Points Insertion

**Your Task:** Connect to Qdrant using **native client**, create collection with explicit vector configs, and upload points using `upload_points`:

```python
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid

def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.
    
    Uses FastEmbed for generating embeddings locally.
    """
    # TODO: Initialize Native Qdrant Client
    client = QdrantClient(url=___)
    
    # TODO: Initialize embedding models (FastEmbed)
    dense_embedding_model = ___  # e.g., "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = ___  # e.g., "Qdrant/bm25" or "prithivida/Splade_PP_en_v1"
    
    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    # TODO: Create collection with explicit vector configurations
    collection_name = ___
    
    # Get embedding dimensions by encoding a sample text
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)
    
    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name=___):
        client.create_collection(
            collection_name=___,
            vectors_config={
                "dense": models.VectorParams(
                    size=___,
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
    texts = [chunk.page_content for chunk in ___]
    
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
        collection_name=___,
        points=___,
        batch_size=___,      # e.g., 64 (default)
        parallel=___,        # e.g., 2 workers
        max_retries=___,     # e.g., 3
        wait=False           # Async mode for better performance
    )
    
    return client, collection_name, dense_embedder, sparse_embedder
```

---

## 🔍 Task 9: Native Hybrid Search with RRF Fusion

**Your Task:** Implement hybrid search using native Qdrant `prefetch` + `FusionQuery`:

```python
def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    """Perform hybrid search using dense + sparse vectors with RRF fusion.
    
    Uses the native Qdrant Query API with prefetch and RRF fusion.
    """
    # TODO: Generate embeddings for the query
    dense_query = list(dense_embedder.embed([___]))[0]
    sparse_query = list(sparse_embedder.embed([___]))[0]
    
    # Convert sparse vector
    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()
    
    # TODO: Perform hybrid search with RRF fusion using prefetch
    results = client.query_points(
        collection_name=___,
        prefetch=[
            # Sparse vector prefetch (keyword search)
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                ),
                using="sparse",
                limit=20  # Retrieve top 20 from sparse search
            ),
            # Dense vector prefetch (semantic search)
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense",
                limit=20  # Retrieve top 20 from dense search
            )
        ],
        # TODO: Apply RRF fusion to combine results
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
        ),
        with_payload=True,
        limit=___  # Final number of results to return
    )
    
    # TODO: Extract and return the documents
    documents = []
    for point in results.points:
        doc = {
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        }
        documents.append(doc)
    
    return documents
```

---

## 🤖 Task 10: Create the Agentic RAG Agent

This is the core of your Agentic RAG system. The agent will intelligently decide when to retrieve documents.

**Your Task:** Create the agent with a retrieval tool using **2026 syntax**:

```python
from langchain.agents import create_agent  # 2026 syntax
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  # For conversation memory

def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """Create an agentic RAG agent with native Qdrant hybrid search tool."""
    
    # TODO: Define the retrieval tool using native Qdrant hybrid search
    @tool(response_format=___)  # Use "content_and_artifact" for best results
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.
        
        Args:
            query: The search query to find relevant documents
        """
        # TODO: Call the hybrid_search_rrf function
        docs = hybrid_search_rrf(
            client=___,
            collection_name=___,
            query_text=___,
            dense_embedder=___,
            sparse_embedder=___,
            limit=___
        )
        
        # TODO: Format the retrieved documents
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs
    
    # TODO: Define the system prompt
    system_prompt = """
    You are a helpful AI assistant with access to a document knowledge base.
    
    Instructions:
    - Use the retrieve_context tool when you need information from the documents
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results
    - Always cite your sources when using retrieved information
    - If the retrieved context doesn't contain relevant information, say "I don't have enough information to answer that question"
    - You can ask follow-up questions if the query is unclear
    """
    
    # TODO: Create the agent using create_agent (2026 syntax)
    # The model string "ollama:<model>" is used directly — no separate ChatOllama import needed
    agent = create_agent(
        model=___,           # e.g., "ollama:qwen3:4b-instruct" (provider:model string)
        tools=___,           # List containing the retrieve_context tool
        system_prompt=___,   # System instructions
        checkpointer=___,    # InMemorySaver() for conversation persistence
    )
    
    return agent
```

---

## 🔗 Task 11: Streaming with Agent (2026 Recommended)

**Your Task:** Stream the agent's response using **2026 `stream_mode="values"` syntax**:

```python
def stream_agent_response(agent, user_query, thread_id="default"):
    """Stream the agent response with stream_mode='values' (2026 recommended approach).
    
    Args:
        agent: The created agent
        user_query: The user's question
        thread_id: Conversation thread ID for persistence
    """
    from langchain.messages import AIMessage, HumanMessage
    
    # TODO: Prepare the input messages (2026 format)
    inputs = {
        "messages": [{"role": "user", "content": ___}]
    }
    
    # TODO: Prepare config with thread_id for conversation memory
    config = {
        "configurable": {
            "thread_id": ___  # Required for checkpointer to work
        }
    }
    
    # TODO: Stream with stream_mode="values" (2026 recommended approach)
    # Each chunk contains the full state at that point
    for chunk in agent.stream(
        ___,
        stream_mode=___,     # "values" for full state at each step
        config=___
    ):
        # Access the latest message from the state
        latest_message = chunk["messages"][-1]
        
        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            # Agent is calling a tool
            yield f"\n🔍 Searching: {[tc['name'] for tc in latest_message.tool_calls]}\n"
```

---

## 🖥️ Task 12: Build the Streamlit UI with Streaming

**Your Task:** Build the complete Streamlit interface:

```python
import streamlit as st
import os
from qdrant_client import QdrantClient

# TODO: Import your functions
# from your_module import (
#     load_and_split_pdf, 
#     setup_qdrant_collection, 
#     create_rag_agent, 
#     stream_agent_response
# )

# TODO: Create page title
st.title(___)

# TODO: Initialize session state for agent and Qdrant components
if "agent" not in st.session_state:
    st.session_state.agent = None
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "dense_embedder" not in st.session_state:
    st.session_state.dense_embedder = None
if "sparse_embedder" not in st.session_state:
    st.session_state.sparse_embedder = None

# TODO: Create Sidebar for File Upload
uploaded_file = st.sidebar.file_uploader(___)

if uploaded_file:
    # TODO: Save file locally
    temp_path = os.path.join(___, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # TODO: Process and store document
    with st.spinner("Processing document with native Qdrant..."):
        # Load and split PDF
        chunks = ___(___)
        
        # Setup native Qdrant collection with dense + sparse vectors
        qdrant_client, collection_name, dense_embedder, sparse_embedder = ___(___)
        
        st.session_state.qdrant_client = qdrant_client
        st.session_state.collection_name = collection_name
        st.session_state.dense_embedder = dense_embedder
        st.session_state.sparse_embedder = sparse_embedder
        
        # TODO: Create the agent with native Qdrant components
        st.session_state.agent = ___(
            qdrant_client,
            collection_name,
            dense_embedder,
            sparse_embedder
        )
    
    st.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")

# TODO: Create Chat Interface
user_input = st.chat_input(___)

if user_input and st.session_state.agent:
    # TODO: Display user message
    st.chat_message("user").write(___)
    
    # TODO: Display assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # TODO: Stream the response using agent.stream (synchronous — no async needed)
        for token in stream_agent_response(
            st.session_state.agent,
            user_input,
            thread_id="session_001"  # Use consistent thread_id for memory
        ):
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        
        # Final update without cursor
        response_placeholder.markdown(full_response)
        
elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first!")
```

---

## 🔄 Alternative: Streaming with Multiple Modes (Advanced)

For better visibility into the agent's reasoning using different stream modes:

```python
def stream_with_updates(agent, user_query):
    """Stream with stream_mode='values' for full visibility into agent steps."""
    from langchain.messages import AIMessage, HumanMessage, ToolMessage
    
    config = {"configurable": {"thread_id": "session_001"}}
    
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_query}]},
        stream_mode="values",
        config=config
    ):
        latest_message = chunk["messages"][-1]
        
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            if latest_message.content:
                print(f"Agent: {latest_message.content}")
            if latest_message.tool_calls:
                print(f"\n[Tool Call] {[tc['name'] for tc in latest_message.tool_calls]}")
        elif isinstance(latest_message, ToolMessage):
            print(f"[Tool Result] {latest_message.content[:200]}...")
```

---

## 🚀 Run the Application

```bash
uv run streamlit run app.py
```
## 🔗 Resources

- [Qdrant Hybrid Queries Documentation](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Hybrid Search with Reranking Tutorial](https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/)
- [Qdrant Points Management](https://qdrant.tech/documentation/manage-data/points/)
- [Build a Hybrid Search API](https://qdrant.tech/documentation/tutorials-develop/hybrid-search-fastembed/)
- [LangChain Agents Documentation](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Event Streaming](https://docs.langchain.com/oss/python/langchain/event-streaming)
- [LangChain Streaming](https://docs.langchain.com/oss/python/langchain/streaming)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Qdrant Python Client v1.18.0](https://pypi.org/project/qdrant-client/)
- [LangChain v1.3.9](https://pypi.org/project/langchain/)
- [LangGraph v1.2.5](https://pypi.org/project/langgraph/)
