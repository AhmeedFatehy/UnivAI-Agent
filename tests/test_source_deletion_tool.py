from document_processing.metadata import stable_document_id
import mcp_server


def test_collection_delete_resolves_ingestions_canonical_document_id(monkeypatch) -> None:
    calls = []

    def delete(user_id: str, document_id: str) -> int:
        calls.append((user_id, document_id))
        return 7

    monkeypatch.setattr(mcp_server, "delete_document", delete)

    message = mcp_server._remove_collection_document_sync(
        "student-a",
        "42",
        "privacy.pdf",
    )

    assert calls == [
        ("student-a", stable_document_id("42", "privacy.pdf")),
    ]
    assert "Removed 7 chunks" in message
