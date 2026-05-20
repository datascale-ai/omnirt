from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np

import pytest

from omnirt.models.wav2lip import loader as wav2lip_loader
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
        wav2lip_postprocess_mode="opentalking_improved",
    )

    state = runtime._session_state(session)

    assert len(state.prepared_frames) == 2
    assert [item["animation"]["mouth_center"] for item in seen] == [[0.25, 0.5], [0.75, 0.5]]


def test_easy_improved_postprocess_mode_prepares_geometry_without_enhanced_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OMNIRT_WAV2LIP_POSTPROCESS_MODE", "easy_improved")
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": {
                    "frame_00000.jpg": {
                        "animation": {
                            "mouth_center": [0.5, 0.5],
                            "mouth_rx": 0.1,
                            "mouth_ry": 0.05,
                            "outer_lip": [
                                [0.4, 0.5],
                                [0.45, 0.46],
                                [0.5, 0.45],
                                [0.55, 0.46],
                                [0.6, 0.5],
                                [0.55, 0.54],
                                [0.5, 0.55],
                                [0.45, 0.54],
                            ],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = Wav2LipRealtimeRuntime(device="cpu")
    monkeypatch.setattr(runtime, "_model_bundle", lambda: {"input_size": 8})
    monkeypatch.setattr(runtime, "_detect_face_box", lambda frame: (0, frame.shape[0], 0, frame.shape[1]))

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
        wav2lip_postprocess_mode="easy_improved",
    )

    state = runtime._session_state(session)

    assert state.prepared_frames[0].geometry is not None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "easy_improved"),
        ("basic", "basic"),
        ("opentalking_improved", "opentalking_improved"),
        ("easy_improved", "easy_improved"),
        ("easy_enhanced", "easy_enhanced"),
        ("unsupported", "easy_improved"),
    ],
)
def test_wav2lip_postprocess_mode_accepts_only_public_values(raw: str | None, expected: str) -> None:
    assert Wav2LipRealtimeRuntime._parse_postprocess_mode(raw) == expected


def test_runtime_defaults_face_detection_to_cpu_for_npu(monkeypatch) -> None:
    monkeypatch.setenv("OMNIRT_WAV2LIP_DEVICE", "npu:0")
    monkeypatch.delenv("OMNIRT_WAV2LIP_FACE_DET_DEVICE", raising=False)

    runtime = Wav2LipRealtimeRuntime()

    assert runtime.device == "npu:0"
    assert runtime.face_detection_device == "cpu"


def test_runtime_uses_explicit_face_detection_device(monkeypatch) -> None:
    monkeypatch.setenv("OMNIRT_WAV2LIP_DEVICE", "cuda")
    monkeypatch.setenv("OMNIRT_WAV2LIP_FACE_DET_DEVICE", "cpu")

    runtime = Wav2LipRealtimeRuntime()

    assert runtime.device == "cuda"
    assert runtime.face_detection_device == "cpu"


def test_runtime_defaults_cpu_thread_limits(monkeypatch) -> None:
    for key in (
        "OMNIRT_WAV2LIP_CPU_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        monkeypatch.delenv(key, raising=False)

    Wav2LipRealtimeRuntime._configure_cpu_thread_limits()

    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "4"
    assert os.environ["OPENCV_FOR_THREADS_NUM"] == "4"


def test_wav2lip_auto_device_uses_configured_npu_index(monkeypatch) -> None:
    class FakeNpu:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        npu = FakeNpu()
        cuda = FakeCuda()

    monkeypatch.setenv("OMNIRT_WAV2LIP_NPU_INDEX", "3")
    monkeypatch.setattr(wav2lip_loader, "_try_import_torch_npu", lambda: True)

    assert wav2lip_loader._resolve_torch_device(FakeTorch, "auto") == "npu:3"
    assert wav2lip_loader._resolve_torch_device(FakeTorch, "npu") == "npu:3"


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
            face_crop=np.zeros((8, 8, 3), dtype=np.uint8),
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
            wav2lip_postprocess_mode="opentalking_improved",
        )

    first = runtime._session_state(make_session("s1"))
    second = runtime._session_state(make_session("s2"))

    assert calls == [0, 1]
    assert first.prepared_frames is second.prepared_frames


def test_frame_sequence_preparation_is_persisted_to_disk(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    cache_dir = tmp_path / "cache"

    first_runtime = Wav2LipRealtimeRuntime(device="cpu")
    first_runtime.prepared_cache_dir = cache_dir

    def fake_prepare(session, frame, *, frame_index, mouth_metadata=None):
        del session, frame_index, mouth_metadata
        return _PreparedFrame(
            base_frame=frame,
            face_crop=np.ones((8, 8, 3), dtype=np.uint8),
            coords=(1, 2, 3, 4),
            geometry=None,
        )

    monkeypatch.setattr(first_runtime, "_prepare_reference_frame", fake_prepare)

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
            wav2lip_postprocess_mode="opentalking_improved",
        )

    first = first_runtime._session_state(make_session("s1"))

    second_runtime = Wav2LipRealtimeRuntime(device="cpu")
    second_runtime.prepared_cache_dir = cache_dir
    monkeypatch.setattr(
        second_runtime,
        "_prepare_reference_frame",
        lambda *args, **kwargs: pytest.fail("prepared disk cache should avoid frame preparation"),
    )
    second = second_runtime._session_state(make_session("s2"))

    assert first.prepared_frames[0].coords == (1, 2, 3, 4)
    assert second.prepared_frames[0].coords == (1, 2, 3, 4)
    assert list(cache_dir.glob("v3-*.npz"))


def test_frame_sequence_preparation_is_shared_across_concurrent_calls(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)

    runtime = Wav2LipRealtimeRuntime(device="cpu")
    calls = 0

    def fake_prepare(session, frame, *, frame_index, mouth_metadata=None):
        nonlocal calls
        del session, frame_index, mouth_metadata
        calls += 1
        time.sleep(0.05)
        return _PreparedFrame(
            base_frame=frame,
            face_crop=np.ones((8, 8, 3), dtype=np.uint8),
            coords=(1, 2, 3, 4),
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
            wav2lip_postprocess_mode="opentalking_improved",
        )

    results: list[list[_PreparedFrame]] = []
    threads = [
        threading.Thread(target=lambda session_id=session_id: results.append(runtime._prepare_frame_sequence(make_session(session_id))))
        for session_id in ("s1", "s2")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_prepared_disk_cache_round_trips_mouth_geometry(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "frame_00000.jpg", 10)
    cache_dir = tmp_path / "cache"

    first_runtime = Wav2LipRealtimeRuntime(device="cpu")
    first_runtime.prepared_cache_dir = cache_dir
    geometry = first_runtime._fallback_mouth_geometry(np.full((8, 8, 3), 10, dtype=np.uint8))

    monkeypatch.setattr(
        first_runtime,
        "_prepare_reference_frame",
        lambda session, frame, *, frame_index, mouth_metadata=None: _PreparedFrame(
            base_frame=frame,
            face_crop=np.ones((8, 8, 3), dtype=np.uint8),
            coords=(1, 2, 3, 4),
            geometry=geometry,
        ),
    )

    session = RealtimeAvatarSession(
        session_id="s1",
        trace_id="t",
        model="wav2lip",
        backend="test",
        prompt="",
        image_bytes=b"ref",
        reference_mode="frames",
        ref_frame_dir=str(frames),
        audio=AvatarAudioSpec(),
        video=AvatarVideoSpec(width=24, height=24),
        wav2lip_postprocess_mode="opentalking_improved",
    )
    first_runtime._session_state(session)

    second_runtime = Wav2LipRealtimeRuntime(device="cpu")
    second_runtime.prepared_cache_dir = cache_dir
    second = second_runtime._session_state(
        RealtimeAvatarSession(
            session_id="s2",
            trace_id="t",
            model="wav2lip",
            backend="test",
            prompt="",
            image_bytes=b"ref",
            reference_mode="frames",
            ref_frame_dir=str(frames),
            audio=AvatarAudioSpec(),
            video=AvatarVideoSpec(width=24, height=24),
            wav2lip_postprocess_mode="opentalking_improved",
        )
    )

    assert second.prepared_frames[0].geometry == geometry


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
            face_crop=np.zeros((8, 8, 3), dtype=np.uint8),
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
            wav2lip_postprocess_mode="opentalking_improved",
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
        wav2lip_postprocess_mode="opentalking_improved",
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
        wav2lip_postprocess_mode="opentalking_improved",
        preprocessed=True,
    )

    state = runtime._session_state(session)

    assert detector_calls == 1
    assert state.prepared_frames[0].coords == (0, 24, 0, 24)


def test_preprocessed_frame_metadata_with_face_box_uses_detector_for_model_crop(tmp_path: Path, monkeypatch) -> None:
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
                        "face_box": [0.25, 0.125, 0.75, 0.875],
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
        wav2lip_postprocess_mode="opentalking_improved",
        preprocessed=True,
    )

    state = runtime._session_state(session)

    assert detector_calls == 1
    assert state.prepared_frames[0].coords == (0, 24, 0, 24)


def test_preprocessed_frame_metadata_with_asset_tuned_crop_skips_detector(tmp_path: Path, monkeypatch) -> None:
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
                        "model_crop": [0.25, 0.125, 0.75, 0.875],
                        "model_crop_source": "asset_tuned",
                        "animation": {"mouth_center": [0.5, 0.5]},
                    }
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
        lambda frame: pytest.fail("asset_tuned model_crop must skip detector"),
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
        wav2lip_postprocess_mode="opentalking_improved",
        preprocessed=True,
    )

    state = runtime._session_state(session)

    assert state.prepared_frames[0].coords == (3, 21, 6, 18)


def test_image_reference_with_asset_tuned_crop_skips_detector(monkeypatch) -> None:
    frame = np.full((24, 24, 3), 10, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok

    runtime = Wav2LipRealtimeRuntime(device="cpu")
    monkeypatch.setattr(runtime, "_model_bundle", lambda: {"input_size": 8})
    monkeypatch.setattr(
        runtime,
        "_detect_face_box",
        lambda frame: pytest.fail("asset_tuned model_crop must skip detector"),
    )
    monkeypatch.setattr(runtime, "_fallback_mouth_geometry", lambda face: None)

    session = RealtimeAvatarSession(
        session_id="s",
        trace_id="t",
        model="wav2lip",
        backend="test",
        prompt="",
        image_bytes=encoded.tobytes(),
        reference_mode="image",
        mouth_metadata={
            "model_crop": [0.25, 0.125, 0.75, 0.875],
            "model_crop_source": "asset_tuned",
            "animation": {"mouth_center": [0.5, 0.5]},
        },
        audio=AvatarAudioSpec(),
        video=AvatarVideoSpec(width=24, height=24),
        wav2lip_postprocess_mode="opentalking_improved",
        preprocessed=False,
    )

    state = runtime._session_state(session)

    assert state.prepared_frames[0].coords == (3, 21, 6, 18)


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
        wav2lip_postprocess_mode="opentalking_improved",
        preprocessed=True,
    )

    with pytest.raises(Wav2LipRuntimeError, match="hash"):
        runtime._session_state(session)
