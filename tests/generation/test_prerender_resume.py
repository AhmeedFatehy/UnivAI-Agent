from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


def load_prerender_module():
    live_dir = Path(__file__).resolve().parents[3] / "UnivAI-live"
    sys.path.insert(0, str(live_dir))
    spec = importlib.util.spec_from_file_location(
        "resumable_prerender_audio", live_dir / "prerender_audio.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_audio_retry_reuses_valid_clips(tmp_path, monkeypatch):
    prerender = load_prerender_module()
    monkeypatch.setattr(prerender, "LECTURES_DIR", tmp_path)
    folder = tmp_path / "student-1" / "week-1"
    folder.mkdir(parents=True)
    (folder / "script.json").write_text(
        json.dumps(
            {
                "title": "Checkpointed audio",
                "segments": [{"text": "First sentence. Second sentence."}],
            }
        ),
        encoding="utf-8",
    )

    class Engine:
        sample_rate = 22050

        def __init__(self):
            self.calls = 0

        def render(self, _sentence):
            self.calls += 1
            return np.ones(32, dtype=np.float32)

    first_engine = Engine()
    monkeypatch.setitem(
        sys.modules,
        "tts",
        types.SimpleNamespace(load_engine=lambda: first_engine),
    )
    first = prerender.prerender_all("student-1", 1, log=lambda _message: None)
    assert first["clips"] == 2
    assert first_engine.calls == 2

    audio = folder / "audio"
    (audio / "meta.json").unlink()
    (audio / "s0-t1.npy").unlink()
    resumed_engine = Engine()
    monkeypatch.setitem(
        sys.modules,
        "tts",
        types.SimpleNamespace(load_engine=lambda: resumed_engine),
    )
    resumed = prerender.prerender_all("student-1", 1, log=lambda _message: None)

    assert resumed["clips"] == 1
    assert resumed["reused"] == 1
    assert resumed_engine.calls == 1
    meta = json.loads((audio / "meta.json").read_text(encoding="utf-8"))
    assert meta["script_sha256"]
