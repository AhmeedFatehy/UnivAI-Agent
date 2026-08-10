import re
from collections.abc import Sequence

from langchain.agents import create_agent  # 2026 syntax
from langchain_core.tools import StructuredTool
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

_TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_READ_ONLY_TOOLS = frozenset({"retrieve_grounded_context", "get_source_location"})


def _validated_tenant_id(user_id: str) -> str:
    value = (user_id or "").strip()
    if not _TENANT_ID.fullmatch(value):
        raise ValueError("user_id must be a valid authenticated tenant identifier")
    return value


def bind_read_only_tools(tools: Sequence[object], user_id: str) -> list[StructuredTool]:
    """Expose only tenant-bound read tools; the model never chooses ``user_id``.

    Upload, listing, deletion, planning, and server-administration tools are not
    available in a question-answering turn. This is an authorization boundary,
    not a prompt suggestion.
    """
    tenant_id = _validated_tenant_id(user_id)
    by_name = {
        getattr(tool, "name", ""): tool
        for tool in tools
        if getattr(tool, "name", "") in _READ_ONLY_TOOLS
    }

    grounded = by_name.get("retrieve_grounded_context")
    locate = by_name.get("get_source_location")
    if grounded is None or locate is None:
        missing = sorted(_READ_ONLY_TOOLS - set(by_name))
        raise RuntimeError(f"required read-only MCP tools are unavailable: {missing}")

    async def retrieve_grounded(
        query: str,
        collection_id: str | None = None,
        document_ids: list[str] | None = None,
        limit: int = 5,
    ) -> str:
        return await grounded.ainvoke(
            {
                "query": query,
                "user_id": tenant_id,
                "collection_id": collection_id,
                "document_ids": document_ids or [],
                "limit": limit,
            }
        )

    async def get_location(
        document_id: str,
        chunk_index: int | None = None,
    ) -> str:
        return await locate.ainvoke(
            {
                "user_id": tenant_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
            }
        )

    return [
        StructuredTool.from_function(
            coroutine=retrieve_grounded,
            name="retrieve_grounded_context",
            description=(
                "Retrieve cited course passages for the authenticated learner. "
                "The tenant identity is bound by the server and is not an argument."
            ),
        ),
        StructuredTool.from_function(
            coroutine=get_location,
            name="get_source_location",
            description=(
                "Resolve one cited document/chunk for the authenticated learner. "
                "This tool is read-only and tenant-bound."
            ),
        ),
    ]

async def run_agent_stream(user_query: str, *, user_id: str, thread_id="session_001"):
    from guardrails.input import classify_user_input

    decision = classify_user_input(user_query)
    if not decision.safe:
        yield (
            "REFUSED: this request looks like a prompt-injection attempt "
            f"(matched: {', '.join(decision.matched_rules)}). The request is not "
            "allowed to override the assistant's instructions, tools or grounding."
        )
        return

    tenant_id = _validated_tenant_id(user_id)
    template = load_prompt_for(PromptOperation.RETRIEVAL_ANSWER)
    system_prompt = template.render_system() + (
        "\n\nThe runtime has tenant-bound the available read tools to learner "
        f"{tenant_id!r}. Never ask for, infer, or accept another tenant id."
    )

    # In langchain-mcp-adapters 0.1.0+, MultiServerMCPClient manages its own connections 
    # per tool call, so it cannot be used as a context manager.
    client = build_mcp_client()
    tools = bind_read_only_tools(await client.get_tools(), tenant_id)
    
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
