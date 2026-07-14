"""Streamlit app for testing the RAG agent and MCP server."""
import asyncio
import os
import streamlit as st
import tempfile
import uuid

from agent import run_agent_stream
from ingest import ingest

st.title("🎓 Agentic RAG Platform Demo")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    # Simulate a logged-in user
    user_id = st.text_input("Simulated User ID", value="student_124")
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload course material", type=["pdf", "docx", "txt", "md"])
    
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Processing and indexing..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = tmp.name
                
                try:
                    # Run direct ingestion (could also use the MCP tool)
                    ingest(temp_path, user_id)
                    st.success(f"Successfully ingested {uploaded_file.name} for {user_id}!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    os.unlink(temp_path)

# Remove caching of agent since it ties to a closed event loop
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat interface
user_input = st.chat_input("Ask a question about your uploaded materials...")

if user_input:
    # Add context about who the user is to the prompt
    contextualized_input = f"[My User ID is '{user_id}'].\nUser Question: {user_input}"
    
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        async def run_stream():
            global full_response
            try:
                # Stream response while managing MCP client lifecycle internally
                async for token in run_agent_stream(
                    contextualized_input,
                    thread_id=st.session_state.session_id
                ):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
            except Exception as e:
                full_response = f"Error: {e}"
                response_placeholder.markdown(full_response)

        with st.spinner("Thinking..."):
            asyncio.run(run_stream())
            
        response_placeholder.markdown(full_response)
        
        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})