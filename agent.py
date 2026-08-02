from langchain.agents import create_agent  # 2026 syntax
from langgraph.checkpoint.memory import InMemorySaver  # For conversation memory
from langchain.agents import create_agent          # 2026 syntax
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from agents.prompts import PromptOperation, load_prompt_for

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
    from guardrails.input import classify_user_input

    decision = classify_user_input(user_query)
    if not decision.safe:
        yield (
            "REFUSED: this request looks like a prompt-injection attempt "
            f"(matched: {', '.join(decision.matched_rules)}). The request is not "
            "allowed to override the assistant's instructions, tools or grounding."
        )
        return

    client = build_mcp_client()

    system_prompt = load_prompt_for(PromptOperation.RETRIEVAL_ANSWER).system
    
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
