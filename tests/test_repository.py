from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_agent_modules_are_present() -> None:
    expected = [
        ROOT / "agent.py",
        ROOT / "mcp_server.py",
        ROOT / "document_processing" / "chunking.py",
        ROOT / "retrieval" / "pipeline.py",
        ROOT / "vector_store" / "collection_manager.py",
    ]

    assert all(path.is_file() for path in expected)
