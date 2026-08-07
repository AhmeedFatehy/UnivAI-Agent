from __future__ import annotations

import json

import numpy as np
import pytest

from generation import lecture_gen


@pytest.fixture
def lectures(tmp_path, monkeypatch):
    monkeypatch.setattr(lecture_gen, "LECTURES_DIR", tmp_path)
    return tmp_path


def voiced_week(root, sid: str, week: int, clips: int = 3):
    audio = root / sid / f"week-{week}" / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    for index in range(clips):
        np.save(audio / f"s0-t{index}.npy", np.ones(8, dtype=np.float32))
    (audio / "meta.json").write_text(
        json.dumps({"sample_rate": 22050, "script_sha256": "abc"}), encoding="utf-8"
    )
    return audio


def test_a_voiced_week_arrives_voiced_without_re_rendering(lectures):
    donor_audio = voiced_week(lectures, "donor", 1)

    assert lecture_gen.adopt_week_audio("donor", "adopter", 1) is True

    adopter_audio = lectures / "adopter" / "week-1" / "audio"
    assert sorted(p.name for p in adopter_audio.glob("*.npy")) == [
        "s0-t0.npy",
        "s0-t1.npy",
        "s0-t2.npy",
    ]
    assert (adopter_audio / "meta.json").is_file()
    assert lecture_gen.valid_audio_checkpoint("adopter", 1)
    # The whole point: one copy on disk under two names. 86 MB a week means
    # copying would cost gigabytes across a cohort.
    assert (donor_audio / "s0-t0.npy").stat().st_ino == (
        adopter_audio / "s0-t0.npy"
    ).stat().st_ino


def test_a_learner_who_re_renders_never_rewrites_the_donors_audio(lectures):
    donor_audio = voiced_week(lectures, "donor", 1)
    lecture_gen.adopt_week_audio("donor", "adopter", 1)
    adopter_clip = lectures / "adopter" / "week-1" / "audio" / "s0-t0.npy"

    # Exactly how prerender_audio writes a clip: render to a temp name, then
    # replace(). That swaps in a new inode rather than writing through the
    # link, which is what makes sharing safe.
    temporary = adopter_clip.with_name(f".{adopter_clip.name}.tmp.npy")
    np.save(temporary, np.full(8, 2.0, dtype=np.float32))
    temporary.replace(adopter_clip)

    assert np.load(adopter_clip)[0] == 2.0
    assert np.load(donor_audio / "s0-t0.npy")[0] == 1.0
    assert adopter_clip.stat().st_ino != (donor_audio / "s0-t0.npy").stat().st_ino


def test_removing_one_learners_clip_leaves_the_other_intact(lectures):
    donor_audio = voiced_week(lectures, "donor", 1)
    lecture_gen.adopt_week_audio("donor", "adopter", 1)

    (lectures / "adopter" / "week-1" / "audio" / "s0-t0.npy").unlink()

    assert (donor_audio / "s0-t0.npy").is_file()
    assert np.load(donor_audio / "s0-t0.npy")[0] == 1.0


def test_a_week_the_donor_never_voiced_is_not_claimed_as_ready(lectures):
    (lectures / "donor" / "week-2").mkdir(parents=True)

    assert lecture_gen.adopt_week_audio("donor", "adopter", 2) is False
    assert not lecture_gen.valid_audio_checkpoint("adopter", 2)


def test_a_half_written_donor_week_is_refused(lectures):
    # meta.json present, no clips: prerender was interrupted. Adopting this
    # would report a voiced week that plays nothing.
    audio = lectures / "donor" / "week-3" / "audio"
    audio.mkdir(parents=True)
    (audio / "meta.json").write_text(json.dumps({"sample_rate": 22050}), encoding="utf-8")

    assert lecture_gen.adopt_week_audio("donor", "adopter", 3) is False
