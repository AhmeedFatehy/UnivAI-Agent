"""Small deterministic token store used only by explicit standalone mode."""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path

from runtime import REPOSITORY_ROOT, ensure_within, standalone_root

TOKEN_RE = re.compile(r"[a-z0-9]+")
STORE_VERSION = 1


def _store_file() -> Path:
    return standalone_root() / "data" / "store.json"


def _load() -> dict:
    path = _store_file()
    if not path.exists():
        return {"version": STORE_VERSION, "documents": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    path = _store_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _safe_input(file_path: str) -> Path:
    path = Path(file_path).expanduser().resolve()
    allowed_roots = [REPOSITORY_ROOT / "fixtures", standalone_root() / "uploads"]
    if not any(path == root.resolve() or root.resolve() in path.parents for root in allowed_roots):
        raise ValueError(
            "Standalone ingest accepts files only from fixtures/ or "
            f"{standalone_root() / 'uploads'}"
        )
    if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
        raise ValueError("Standalone ingest supports project-authored text/Markdown fixtures")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _chunks(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > 700:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def ingest(file_path: str, user_id: str) -> dict:
    if not user_id.strip():
        raise ValueError("user_id is required")
    path = _safe_input(file_path)
    content = path.read_text(encoding="utf-8")
    parts = _chunks(content)
    if not parts:
        raise ValueError("document is empty")

    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"univai:{user_id}:{path.name}"))
    data = _load()
    data["documents"] = [
        item
        for item in data["documents"]
        if not (item["user_id"] == user_id and item["document_id"] == document_id)
    ]
    data["documents"].append(
        {
            "document_id": document_id,
            "user_id": user_id,
            "source_filename": path.name,
            "document_type": path.suffix.lstrip("."),
            "upload_date": "2026-07-27T00:00:00+00:00",
            "chunks": [
                {
                    "content": chunk,
                    "chunk_index": index,
                    "page_number": index + 1,
                }
                for index, chunk in enumerate(parts)
            ],
        }
    )
    _save(data)
    return {"document_id": document_id, "chunks_indexed": len(parts)}


def list_documents(user_id: str) -> list[dict]:
    return [
        {
            key: item[key]
            for key in (
                "document_id",
                "source_filename",
                "document_type",
                "upload_date",
            )
        }
        for item in _load()["documents"]
        if item["user_id"] == user_id
    ]


def retrieve(query: str, user_id: str, limit: int = 5) -> str:
    query_tokens = _tokens(query)
    if not query_tokens:
        return "No relevant documents found."

    ranked: list[tuple[float, dict, dict]] = []
    for document in _load()["documents"]:
        if document["user_id"] != user_id:
            continue
        for chunk in document["chunks"]:
            chunk_tokens = _tokens(chunk["content"])
            overlap = len(query_tokens & chunk_tokens)
            if not overlap:
                continue
            score = overlap / math.sqrt(len(query_tokens) * max(1, len(chunk_tokens)))
            ranked.append((score, document, chunk))

    ranked.sort(key=lambda item: (-item[0], item[2]["chunk_index"]))
    relevant = [item for item in ranked if item[0] >= 0.08][: max(1, limit)]
    if not relevant:
        return "No relevant documents found."

    parts = []
    for rank, (score, document, chunk) in enumerate(relevant, start=1):
        parts.append(
            f"[{rank}] Source: {document['source_filename']} | "
            f"Page: {chunk['page_number']} | Chunk: {chunk['chunk_index']}/"
            f"{len(document['chunks'])} | Score: {score:.4f}\n"
            f"Content: {chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def remove(user_id: str, document_id: str) -> int:
    data = _load()
    kept = []
    deleted = 0
    for document in data["documents"]:
        if document["user_id"] == user_id and document["document_id"] == document_id:
            deleted += len(document["chunks"])
        else:
            kept.append(document)
    data["documents"] = kept
    _save(data)
    return deleted


def reset() -> None:
    root = ensure_within(standalone_root(), REPOSITORY_ROOT, label="standalone root")
    path = root / "data" / "store.json"
    if path.exists():
        path.unlink()
