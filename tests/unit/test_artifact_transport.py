"""Tests for the inline-bytes artifact transport and the proto wire roundtrip."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from omnirt.core.artifact_transport import (
    inline_limit_bytes,
    pack_artifact,
    unpack_artifact,
)
from omnirt.core.types import Artifact, ArtifactTooLargeError


def _video_artifact(path: Path) -> Artifact:
    return Artifact(
        kind="video",
        path=str(path),
        mime="video/mp4",
        width=416,
        height=704,
        num_frames=29,
    )


def test_pack_path_is_noop(tmp_path: Path) -> None:
    source = tmp_path / "out.mp4"
    source.write_bytes(b"fake video bytes")
    artifact = _video_artifact(source)
    packed = pack_artifact(artifact, transport="path")
    assert packed.transport == "path"
    assert packed.data_b64 is None
    assert packed.path == str(source)


def test_pack_inline_bytes_embeds_payload(tmp_path: Path) -> None:
    source = tmp_path / "out.mp4"
    payload = b"the quick brown fox " * 1000  # ~20 KB
    source.write_bytes(payload)
    artifact = _video_artifact(source)
    packed = pack_artifact(artifact, transport="inline_bytes")
    assert packed.transport == "inline_bytes"
    assert packed.path == "out.mp4"  # just the filename, not the worker-side path
    assert packed.data_b64 is not None
    assert base64.b64decode(packed.data_b64) == payload
    assert packed.width == 416
    assert packed.num_frames == 29


def test_pack_inline_bytes_raises_when_over_limit(tmp_path: Path) -> None:
    source = tmp_path / "big.mp4"
    source.write_bytes(b"x" * 2048)
    artifact = _video_artifact(source)
    with pytest.raises(ArtifactTooLargeError) as excinfo:
        pack_artifact(artifact, transport="inline_bytes", max_bytes=1024)
    assert excinfo.value.size_bytes == 2048
    assert excinfo.value.max_bytes == 1024
    assert "OMNIRT_ARTIFACT_INLINE_MAX_MB" in str(excinfo.value)


def test_unpack_inline_bytes_writes_to_output_dir(tmp_path: Path) -> None:
    payload = b"hello world" * 50
    inline = Artifact(
        kind="video",
        path="hello.mp4",
        mime="video/mp4",
        width=1,
        height=1,
        num_frames=1,
        transport="inline_bytes",
        data_b64=base64.b64encode(payload).decode("ascii"),
    )
    output_dir = tmp_path / "outputs"
    unpacked = unpack_artifact(inline, output_dir=output_dir)
    assert unpacked.transport == "path"
    assert unpacked.data_b64 is None
    assert Path(unpacked.path).read_bytes() == payload
    assert Path(unpacked.path).parent == output_dir


def test_unpack_path_is_noop(tmp_path: Path) -> None:
    source = tmp_path / "already.mp4"
    source.write_bytes(b"anything")
    artifact = _video_artifact(source)
    unpacked = unpack_artifact(artifact, output_dir=tmp_path / "outputs")
    assert unpacked is artifact


def test_inline_limit_bytes_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("OMNIRT_ARTIFACT_INLINE_MAX_MB", "1")
    assert inline_limit_bytes() == 1024 * 1024
    monkeypatch.setenv("OMNIRT_ARTIFACT_INLINE_MAX_MB", "128")
    assert inline_limit_bytes() == 128 * 1024 * 1024


def test_inline_limit_bytes_rejects_invalid(monkeypatch) -> None:
    monkeypatch.setenv("OMNIRT_ARTIFACT_INLINE_MAX_MB", "-1")
    with pytest.raises(ValueError, match="must be positive"):
        inline_limit_bytes()
    monkeypatch.setenv("OMNIRT_ARTIFACT_INLINE_MAX_MB", "not-a-number")
    with pytest.raises(ValueError, match="integer"):
        inline_limit_bytes()


# --------------------------------------------------------------- proto roundtrip

def test_artifact_proto_roundtrip_preserves_all_fields(tmp_path: Path) -> None:
    pytest.importorskip("grpc")
    from omnirt.engine.grpc_transport import artifact_from_proto, artifact_to_proto

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"binary content" * 100)
    original = pack_artifact(_video_artifact(source), transport="inline_bytes")
    roundtrip = artifact_from_proto(artifact_to_proto(original))

    assert roundtrip.kind == original.kind
    assert roundtrip.path == original.path
    assert roundtrip.mime == original.mime
    assert roundtrip.width == original.width
    assert roundtrip.height == original.height
    assert roundtrip.num_frames == original.num_frames
    assert roundtrip.transport == "inline_bytes"
    assert roundtrip.data_b64 == original.data_b64


def test_artifact_path_roundtrip() -> None:
    pytest.importorskip("grpc")
    from omnirt.engine.grpc_transport import artifact_from_proto, artifact_to_proto

    original = Artifact(
        kind="video",
        path="/shared/fs/clip.mp4",
        mime="video/mp4",
        width=416,
        height=704,
        num_frames=29,
    )
    roundtrip = artifact_from_proto(artifact_to_proto(original))
    assert roundtrip.transport == "path"
    assert roundtrip.data_b64 is None
    assert roundtrip.path == "/shared/fs/clip.mp4"


def test_request_and_result_proto_roundtrip() -> None:
    pytest.importorskip("grpc")
    from omnirt.core.types import GenerateRequest, GenerateResult, RunReport
    from omnirt.engine.grpc_transport import (
        request_from_proto,
        request_to_proto,
        result_from_proto,
        result_to_proto,
    )

    request = GenerateRequest(
        task="audio2video",
        model="soulx-flashtalk-14b",
        backend="ascend",
        inputs={"image": "/tmp/a.png", "audio": "/tmp/a.wav"},
        config={"seed": 9999, "nproc_per_node": 8},
    )
    req_back = request_from_proto(request_to_proto(request))
    assert req_back.task == request.task
    assert req_back.inputs == request.inputs
    assert req_back.config == request.config

    report = RunReport(
        run_id="r-1",
        task=request.task,
        model=request.model,
        backend=request.backend,
        execution_mode="persistent_worker",
        timings={"denoise_loop_ms": 1234.5},
        memory={"peak_mb": 2048},
        config_resolved={"seed": 9999},
    )
    result = GenerateResult(
        outputs=[
            Artifact(
                kind="video",
                path="/tmp/out.mp4",
                mime="video/mp4",
                width=416,
                height=704,
                num_frames=29,
            )
        ],
        metadata=report,
    )
    result_back = result_from_proto(result_to_proto(result))
    assert result_back.metadata.run_id == "r-1"
    assert result_back.metadata.execution_mode == "persistent_worker"
    assert result_back.metadata.timings == {"denoise_loop_ms": 1234.5}
    assert result_back.outputs[0].path == "/tmp/out.mp4"
    assert result_back.outputs[0].transport == "path"
