from __future__ import annotations

from types import MethodType

import numpy as np
import pytest


def test_quicktalk_auto_device_prefers_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setattr(
        quicktalk_runtime,
        "_is_accelerator_available",
        lambda kind: kind == "npu",
    )

    assert quicktalk_runtime.resolve_quicktalk_device("auto") == "npu:0"


def test_quicktalk_auto_device_prefers_cuda_when_npu_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setattr(
        quicktalk_runtime,
        "_is_accelerator_available",
        lambda kind: kind == "cuda",
    )

    assert quicktalk_runtime.resolve_quicktalk_device("auto") == "cuda:0"


def test_quicktalk_streaming_pcm_features_skip_compressed_audio_cache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.models.quicktalk.runtime_worker import RealtimeV3SessionState, RealtimeV3Worker

    class FakeQuickTalkV2:
        face_cache_dir = tmp_path
        sync_offset = 0

        def extract_representations_pcm(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
            return np.zeros((1, 10, 1024), dtype=np.float32)

        def build_rep_chunks(self, repst: np.ndarray, n_frames: int, fps: float) -> list[np.ndarray]:
            return [np.full((10, 1024), index, dtype=np.float32) for index in range(n_frames)]

    monkeypatch.setenv("OMNIRT_QUICKTALK_STREAMING_LOOKAHEAD_CHUNKS", "0")
    worker = object.__new__(RealtimeV3Worker)
    worker.v2 = FakeQuickTalkV2()
    worker.fps = 25.0
    worker._hubert_cache_identity = MethodType(
        lambda self: {"path": "fake-hubert", "files": [], "device_type": "cpu"},
        worker,
    )

    pcm = np.zeros(2560, dtype=np.int16)
    reps, _elapsed = worker.prepare_streaming_pcm_features(
        pcm,
        16_000,
        state=RealtimeV3SessionState(),
    )

    assert len(reps) == 4
    assert list(tmp_path.glob("audio_pcm_*.npz")) == []
