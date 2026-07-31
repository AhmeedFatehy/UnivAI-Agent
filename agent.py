from langchain.agents import create_agent  # 2026 syntax
from langgraph.checkpoint.memory import InMemorySaver  # For conversation memory
from langchain.agents import create_agent          # 2026 syntax
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

MCP_URL = "http://127.0.0.1:8000/mcp"

# TODO: Configure the MCP client to reach your server over HTTP.
# In langchain-mcp-adapters the streamable-HTTP transport value is "http"
# ("streamable_http" is also accepted as an alias). The server side runs
# mcp.run(transport="streamable-http") — same transport, different spelling.
def build_mcp_client():
    return MultiServerMCPClient(
        {
            "rag": {
                "url": MCP_URL,                 # MCP_URL
                "transport": "http",           # "http"
            }
        }
    )

from langchain.messages import AIMessage

async def run_agent_stream(user_query, thread_id="session_001"):
    client = build_mcp_client()

    system_prompt = """
    You are a helpful AI assistant with access to a document knowledge base
    through MCP tools served over HTTP.

    Instructions:
    - ALWAYS use a retrieval tool when you need information to answer the user's question.
    - Prefer retrieve_grounded_context: it returns either passages with book,
      page and section citations, or an explicit refusal. Fall back to
      retrieve_context for plain text results.
    - IMPORTANT: both retrieval tools require a 'user_id' parameter. You MUST extract this from the user's message (e.g., "[My User ID is 'student_123']") and pass it to the tool.
    - Pass collection_id when the user names a course or programme, so the
      answer is scoped to the right set of books.
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion.
    - Cite every claim with the book title, page and section from the tool result.
      Never write a citation the tool did not give you.
    - If the tool returns "grounded": false, relay its refusal reason. Say
      "I don't have enough information to answer that question" rather than
      answering from your own knowledge.
    - Use get_source_location when the user asks you to verify a citation.
    """
    
    # In langchain-mcp-adapters 0.1.0+, MultiServerMCPClient manages its own connections 
    # per tool call, so it cannot be used as a context manager.
    client = build_mcp_client()
    tools = await client.get_tools()
    
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}
    
    async for chunk in agent.astream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]
        
        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            yield f"\n🔍 Calling MCP tool: {[tc['name'] for tc in latest_message.tool_calls]}\n"