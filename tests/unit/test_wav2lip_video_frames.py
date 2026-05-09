from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

import pytest

from omnirt.models.wav2lip.runtime import Wav2LipRealtimeRuntime, Wav2LipRuntimeError, _PreparedFrame
from omnirt.server.realtime_avatar import (
    AvatarAudioSpec,
    AvatarVideoSpec,
    RealtimeAvatarSession,
)


def _write_frame(path: Path, color: int) -> None:
    frame = np.full((24, 24, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frame_sequence_uses_per_frame_mouth_metadata(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    _write_frame(frames / "frame_00001.jpg", 20)
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": {
                    "frame_00000.jpg": {"animation": {"mouth_center": [0.25, 0.5]}},
                    "frame_00001.jpg": {"animation": {"mouth_center": [0.75, 0.5]}},
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = Wav2LipRealtimeRuntime(device="cpu")
    seen: list[dict] = []
    monkeypatch.setattr(runtime, "_model_bundle", lambda: {"input_size": 8})
    monkeypatch.setattr(runtime, "_detect_face_box", lambda frame: (0, frame.shape[0], 0, frame.shape[1]))

    def fake_geometry(metadata, coords, input_shape, frame_shape):
        del coords, input_shape, frame_shape
        seen.append(metadata)
        return None

    monkeypatch.setattr(runtime, "_geometry_from_metadata", fake_geometry)
    monkeypatch.setattr(runtime, "_fallback_mouth_geometry", lambda face: None)

    session = RealtimeAvatarSession(
        session_id="s",
        trace_id="t",
        model="wav2lip",
        backend="test",
        prompt="",
        image_bytes=b"ref",
        reference_mode="frames",
        ref_frame_dir=str(frames),
        ref_frame_metadata_path=str(metadata_path),
        audio=AvatarAudioSpec(),
        video=AvatarVideoSpec(width=24, height=24),
        enable_enhanced_postprocessing=True,
    )

    state = runtime._session_state(session)

    assert len(state.prepared_frames) == 2
    assert [item["animation"]["mouth_center"] for item in seen] == [[0.25, 0.5], [0.75, 0.5]]


def test_frame_sequence_preparation_is_reused_across_sessions(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    _write_frame(frames / "frame_00001.jpg", 20)
    runtime = Wav2LipRealtimeRuntime(device="cpu")
    calls: list[int] = []

    def fake_prepare(session, frame, *, frame_index, mouth_metadata=None):
        del session, mouth_metadata
        calls.append(frame_index)
        return _PreparedFrame(
            base_frame=frame,
            face_input=np.zeros((8, 8, 6), dtype=np.float32),
            coords=(0, frame.shape[0], 0, frame.shape[1]),
            geometry=None,
        )

    monkeypatch.setattr(runtime, "_prepare_reference_frame", fake_prepare)

    def make_session(session_id: str) -> RealtimeAvatarSession:
        return RealtimeAvatarSession(
            session_id=session_id,
            trace_id="t",
            model="wav2lip",
            backend="test",
            prompt="",
            image_bytes=b"ref",
            reference_mode="frames",
            ref_frame_dir=str(frames),
            audio=AvatarAudioSpec(),
            video=AvatarVideoSpec(width=24, height=24),
            enable_enhanced_postprocessing=True,
        )

    first = runtime._session_state(make_session("s1"))
    second = runtime._session_state(make_session("s2"))

    assert calls == [0, 1]
    assert first.prepared_frames is second.prepared_frames


def test_frame_sequence_cache_ignores_session_mouth_metadata_when_frame_metadata_exists(
    tmp_path: Path, monkeypatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text(
        json.dumps({"frames": {"frame_00000.jpg": {"animation": {"mouth_center": [0.5, 0.5]}}}}),
        encoding="utf-8",
    )
    runtime = Wav2LipRealtimeRuntime(device="cpu")
    calls: list[int] = []

    def fake_prepare(session, frame, *, frame_index, mouth_metadata=None):
        del session, mouth_metadata
        calls.append(frame_index)
        return _PreparedFrame(
            base_frame=frame,
            face_input=np.zeros((8, 8, 6), dtype=np.float32),
            coords=(0, frame.shape[0], 0, frame.shape[1]),
            geometry=None,
        )

    monkeypatch.setattr(runtime, "_prepare_reference_frame", fake_prepare)

    def make_session(session_id: str, mouth_metadata: dict | None) -> RealtimeAvatarSession:
        return RealtimeAvatarSession(
            session_id=session_id,
            trace_id="t",
            model="wav2lip",
            backend="test",
            prompt="",
            image_bytes=b"ref",
            reference_mode="frames",
            ref_frame_dir=str(frames),
            ref_frame_metadata_path=str(metadata_path),
            mouth_metadata=mouth_metadata,
            audio=AvatarAudioSpec(),
            video=AvatarVideoSpec(width=24, height=24),
            enable_enhanced_postprocessing=True,
        )

    preloaded = runtime._session_state(make_session("preload", None))
    actual = runtime._session_state(
        make_session("actual", {"animation": {"mouth_center": [0.25, 0.75]}})
    )

    assert calls == [0]
    assert actual.prepared_frames is preloaded.prepared_frames


def test_preprocessed_frame_metadata_uses_model_crop_without_detector(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": {
                        "frame_00000.jpg": {
                            "source_frame_hash": _sha256(frames / "frame_00000.jpg"),
                            "model_crop": [0.25, 0.25, 0.75, 0.75],
                            "model_crop_source": "wav2lip_detector",
                        "animation": {"mouth_center": [0.5, 0.5], "mouth_rx": 0.1, "mouth_ry": 0.05},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = Wav2LipRealtimeRuntime(device="cpu")
    monkeypatch.setattr(runtime, "_model_bundle", lambda: {"input_size": 8})
    monkeypatch.setattr(
        runtime,
        "_detect_face_box",
        lambda frame: pytest.fail("preprocessed frame metadata must skip detector"),
    )
    monkeypatch.setattr(runtime, "_fallback_mouth_geometry", lambda face: None)

    session = RealtimeAvatarSession(
        session_id="s",
        trace_id="t",
        model="wav2lip",
        backend="test",
        prompt="",
        image_bytes=b"ref",
        reference_mode="frames",
        ref_frame_dir=str(frames),
        ref_frame_metadata_path=str(metadata_path),
        audio=AvatarAudioSpec(),
        video=AvatarVideoSpec(width=24, height=24),
        enable_enhanced_postprocessing=True,
        preprocessed=True,
    )

    state = runtime._session_state(session)

    assert state.prepared_frames[0].coords == (6, 18, 6, 18)


def test_preprocessed_frame_metadata_without_trusted_model_crop_uses_detector(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": {
                    "frame_00000.jpg": {
                        "source_frame_hash": _sha256(frames / "frame_00000.jpg"),
                        "model_crop": [0.25, 0.25, 0.75, 0.75],
                        "animation": {"mouth_center": [0.5, 0.5]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = Wav2LipRealtimeRuntime(device="cpu")
    monkeypatch.setattr(runtime, "_model_bundle", lambda: {"input_size": 8})
    detector_calls = 0

    def fake_detect(frame):
        nonlocal detector_calls
        detector_calls += 1
        return (0, frame.shape[0], 0, frame.shape[1])

    monkeypatch.setattr(runtime, "_detect_face_box", fake_detect)
    monkeypatch.setattr(runtime, "_fallback_mouth_geometry", lambda face: None)

    session = RealtimeAvatarSession(
        session_id="s",
        trace_id="t",
        model="wav2lip",
        backend="test",
        prompt="",
        image_bytes=b"ref",
        reference_mode="frames",
        ref_frame_dir=str(frames),
        ref_frame_metadata_path=str(metadata_path),
        audio=AvatarAudioSpec(),
        video=AvatarVideoSpec(width=24, height=24),
        enable_enhanced_postprocessing=True,
        preprocessed=True,
    )

    state = runtime._session_state(session)

    assert detector_calls == 1
    assert state.prepared_frames[0].coords == (0, 24, 0, 24)


def test_preprocessed_frame_metadata_rejects_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": {
                    "frame_00000.jpg": {
                        "source_frame_hash": "wrong",
                        "model_crop": [0.25, 0.25, 0.75, 0.75],
                        "model_crop_source": "wav2lip_detector",
                        "animation": {"mouth_center": [0.5, 0.5]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = Wav2LipRealtimeRuntime(device="cpu")
    monkeypatch.setattr(runtime, "_model_bundle", lambda: {"input_size": 8})

    session = RealtimeAvatarSession(
        session_id="s",
        trace_id="t",
        model="wav2lip",
        backend="test",
        prompt="",
        image_bytes=b"ref",
        reference_mode="frames",
        ref_frame_dir=str(frames),
        ref_frame_metadata_path=str(metadata_path),
        audio=AvatarAudioSpec(),
        video=AvatarVideoSpec(width=24, height=24),
        enable_enhanced_postprocessing=True,
        preprocessed=True,
    )

    with pytest.raises(Wav2LipRuntimeError, match="hash"):
        runtime._session_state(session)
