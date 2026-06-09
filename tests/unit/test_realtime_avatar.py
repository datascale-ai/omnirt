from __future__ import annotations

import base64
import io
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from omnirt.server import create_app  # noqa: E402
from omnirt.server.realtime_avatar import (  # noqa: E402
    AvatarVideoSpec,
    MAGIC_AUDIO,
    MAGIC_FRAME,
    MAGIC_VIDEO,
    RealtimeAvatarService,
    FakeRealtimeAvatarRuntime,
    RealtimeAvatarError,
    decode_jpeg_sequence,
    encode_jpeg_sequence,
    split_frame_payload,
    _scale_video_to_max_long_edge,
)  # noqa: E402

from omnirt.models.wav2lip.runtime import AvatarRuntimeRouter  # noqa: E402
from omnirt.models.fasterliveportrait import runtime as flp_runtime  # noqa: E402
from omnirt.models.fasterliveportrait.runtime import (  # noqa: E402
    FASTLIVEPORTRAIT_MODEL_ID,
    FasterLivePortraitRuntimeError,
    FasterLivePortraitRealtimeRuntime,
)


def _image_b64() -> str:
    return base64.b64encode(b"fake-image-bytes").decode("ascii")

def _audio_payload(chunk_samples: int) -> bytes:
    return MAGIC_AUDIO + (b"\0\0" * chunk_samples)


def _frame_payload(frame: bytes | None = None) -> bytes:
    return MAGIC_FRAME + (frame if frame is not None else _png_bytes((48, 48)))


def _png_bytes(source: tuple[int, int] | np.ndarray) -> bytes:
    buf = io.BytesIO()
    if isinstance(source, tuple):
        Image.new("RGB", source, (128, 96, 64)).save(buf, format="PNG")
    else:
        Image.fromarray(source.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_video_jpeg_sequence_round_trip() -> None:
    payload = encode_jpeg_sequence([b"jpeg-1", b"jpeg-2"])

    assert payload[:4] == MAGIC_VIDEO
    assert decode_jpeg_sequence(payload) == [b"jpeg-1", b"jpeg-2"]


def test_video_jpeg_sequence_allows_empty_priming_chunk() -> None:
    payload = encode_jpeg_sequence([])

    assert payload[:4] == MAGIC_VIDEO
    assert decode_jpeg_sequence(payload) == []


def test_video_jpeg_sequence_rejects_malformed_frame_length() -> None:
    payload = MAGIC_VIDEO + struct.pack("<I", 1) + struct.pack("<I", 99) + b"tiny"

    with pytest.raises(RealtimeAvatarError) as exc:
        decode_jpeg_sequence(payload)

    assert exc.value.code == "bad_video_chunk"


def test_frame_payload_round_trip() -> None:
    jpeg = _png_bytes((32, 32))

    assert split_frame_payload(MAGIC_FRAME + jpeg) == jpeg


def test_frame_payload_rejects_bad_magic() -> None:
    with pytest.raises(RealtimeAvatarError) as exc:
        split_frame_payload(b"AUDI" + b"not-a-frame")

    assert exc.value.code == "bad_frame_magic"


def test_fasterliveportrait_video_clone_ws_init_frame_and_close() -> None:
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=FasterLivePortraitRealtimeRuntime(load_models=False)
    )
    client = TestClient(app)

    with client.websocket_connect("/v1/video2video/fasterliveportrait") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": base64.b64encode(_png_bytes((64, 64))).decode("ascii"),
                "width": 96,
                "height": 96,
                "fps": 12,
                "flag_crop_driving_video": True,
                "animation_region": "all",
                "driving_multiplier": 1.1,
                "expression_multiplier": 1.2,
                "head_motion_multiplier": 0.8,
            }
        )
        init = ws.receive_json()
        assert init["type"] == "init_ok"
        assert init["model"] == FASTLIVEPORTRAIT_MODEL_ID
        assert init["protocol"] == "video-clone"
        assert init["frame_magic"] == "FRAM"
        assert init["video_magic"] == "VIDX"
        assert init["fps"] == 12

        ws.send_bytes(_frame_payload())
        video = ws.receive_bytes()
        assert video[:4] == MAGIC_VIDEO
        frames = decode_jpeg_sequence(video)
        assert len(frames) == 1

        session_id = next(iter(app.state.realtime_avatar_service._sessions))
        session = app.state.realtime_avatar_service._sessions[session_id]
        assert session.runtime_config["flag_crop_driving_video"] is True
        assert session.runtime_config["flag_stitching"] is True
        assert session.runtime_config["flag_pasteback"] is True
        assert session.runtime_config["head_only_pasteback"] is False
        assert session.runtime_config["driving_multiplier"] == 1.1
        assert session.runtime_config["expression_multiplier"] == 1.2

        ws.send_json({"type": "close"})
        assert ws.receive_json()["type"] == "close_ok"
        assert app.state.realtime_avatar_service._sessions == {}


def test_video_clone_ws_requires_frame_magic() -> None:
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=FasterLivePortraitRealtimeRuntime(load_models=False)
    )
    client = TestClient(app)

    with client.websocket_connect("/v1/avatar/video-clone/fasterliveportrait") as ws:
        ws.send_json({"type": "init", "ref_image": base64.b64encode(_png_bytes((64, 64))).decode("ascii")})
        assert ws.receive_json()["type"] == "init_ok"
        ws.send_bytes(MAGIC_AUDIO + b"not-a-jpeg")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "bad_frame_magic"


def test_audio2video_models_reports_fasterliveportrait_video_clone_runtime() -> None:
    class FakeRouter:
        runtime_kind = "router"
        wav2lip = None
        quicktalk = None
        fasterliveportrait = object()

    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=FakeRouter())
    client = TestClient(app)

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["fasterliveportrait_video_clone"] == {
        "id": "fasterliveportrait_video_clone",
        "connected": True,
        "reason": "fasterliveportrait_runtime",
    }


def test_wav2lip_scaled_video_dimensions_are_h264_safe() -> None:
    video = AvatarVideoSpec(width=830, height=1108, fps=30, slice_len=28)

    scaled = _scale_video_to_max_long_edge(video, 832)

    assert scaled.width % 2 == 0
    assert scaled.height % 2 == 0
    assert scaled.width == 622
    assert scaled.height == 832


def test_flashtalk_compatible_ws_init_generate_and_close() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/flashtalk") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64(), "prompt": "talk", "seed": 1})
        init = ws.receive_json()
        assert init["type"] == "init_ok"
        assert init["fps"] == 25
        assert init["slice_len"] == 28

        ws.send_bytes(_audio_payload(init["slice_len"] * 16000 // init["fps"]))
        video = ws.receive_bytes()
        assert video[:4] == MAGIC_VIDEO
        assert len(decode_jpeg_sequence(video)) == 1

        ws.send_json({"type": "close"})
        assert ws.receive_json()["type"] == "close_ok"


def test_fasterliveportrait_ws_accepts_runtime_config_update() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/fasterliveportrait") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "width": 96,
                "height": 96,
                "chunk_samples": 8000,
                "mouth_open_multiplier": 1.0,
            }
        )
        assert ws.receive_json()["type"] == "init_ok"

        ws.send_json(
            {
                "type": "config_update",
                "config": {
                    "mouth_open_multiplier": 1.8,
                    "pose_motion_multiplier": 0.2,
                    "animation_region": "lip",
                    "width": 999,
                },
            }
        )
        assert ws.receive_json() == {
            "type": "config_ok",
            "updated": {
                "mouth_open_multiplier": 1.8,
                "pose_motion_multiplier": 0.2,
                "animation_region": "lip",
            },
        }


def test_fasterliveportrait_session_preserves_realtime_config() -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)

    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cuda",
        image_bytes=b"fake-image-bytes",
        config={
            "fps": 25,
            "width": 320,
            "height": 320,
            "slice_len": 12,
            "chunk_samples": 8000,
            "head_motion_multiplier": 1.0,
            "pose_motion_multiplier": 0.35,
            "animation_region": "lip",
            "expression_multiplier": 1.2,
            "cfg_scale": 4.0,
            "cfg_cond": [],
            "flag_stitching": False,
            "flag_lip_retargeting": True,
            "lip_retargeting_multiplier": 2.5,
            "lookahead_ms": 320,
            "emit_frames_per_chunk": 12,
        },
    )

    assert session.model == FASTLIVEPORTRAIT_MODEL_ID
    assert session.audio.chunk_samples == 8000
    assert session.video.slice_len == 12
    assert session.runtime_config["head_motion_multiplier"] == 1.0
    assert session.runtime_config["pose_motion_multiplier"] == 0.35
    assert session.runtime_config["animation_region"] == "lip"
    assert session.runtime_config["expression_multiplier"] == 1.2
    assert session.runtime_config["cfg_scale"] == 4.0
    assert session.runtime_config["cfg_cond"] == []
    assert session.runtime_config["flag_stitching"] is False
    assert session.runtime_config["flag_lip_retargeting"] is True
    assert session.runtime_config["lip_retargeting_multiplier"] == 2.5
    assert session.runtime_config["lookahead_ms"] == 320
    assert session.runtime_config["emit_frames_per_chunk"] == 12


def test_fasterliveportrait_runtime_config_can_be_updated_in_place() -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cuda",
        image_bytes=b"fake-image-bytes",
        config={
            "fps": 25,
            "width": 320,
            "height": 320,
            "chunk_samples": 8000,
            "mouth_open_multiplier": 1.0,
        },
    )

    updated = service.update_runtime_config(
        session.session_id,
        {
            "mouth_open_multiplier": 1.8,
            "pose_motion_multiplier": 0.2,
            "width": 999,
        },
    )

    assert updated == {
        "mouth_open_multiplier": 1.8,
        "pose_motion_multiplier": 0.2,
    }
    assert session.runtime_config["mouth_open_multiplier"] == 1.8
    assert session.runtime_config["pose_motion_multiplier"] == 0.2
    assert session.video.width == 320
    assert "width" not in session.runtime_config


def test_fasterliveportrait_runtime_emits_configured_frame_count() -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={"chunk_samples": 8000, "emit_frames_per_chunk": 12, "width": 96, "height": 96},
    )

    payload, metrics = service.push_audio_chunk(session.session_id, _audio_payload(8000))

    assert metrics["type"] == "metrics"
    assert metrics["chunk_index"] == 1
    assert len(decode_jpeg_sequence(payload)) == 12
    assert runtime.session_state(session.session_id).emitted_frames == 12


def test_fasterliveportrait_video_clone_prefers_full_frame_pasteback_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=_png_bytes((48, 72)),
        config={"width": 48, "height": 72, "flag_stitching": True, "flag_pasteback": True},
    )
    state = runtime._session_state(session)
    update_args: dict[str, object] = {}

    class FakePipeline:
        src_imgs = [np.zeros((72, 48, 3), dtype=np.uint8)]
        src_infos = ["src"]
        source_path = None

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            update_args.update(args)

        def run(self, image, img_src, src_info, **kwargs):
            driving_crop = np.zeros((32, 32, 3), dtype=np.uint8)
            animated_crop = np.full((32, 32, 3), (240, 20, 20), dtype=np.uint8)
            pasted_full_frame = np.full((72, 48, 3), (20, 220, 20), dtype=np.uint8)
            return driving_crop, animated_crop, pasted_full_frame, ({}, None, None)

    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"pipeline": FakePipeline()})

    payload = runtime._render_model_driving_frame(session, _png_bytes((64, 64)), state)
    frame = np.asarray(Image.open(io.BytesIO(decode_jpeg_sequence(payload)[0])).convert("RGB"))

    assert update_args["flag_stitching"] is True
    assert update_args["flag_pasteback"] is True
    assert frame.shape == (72, 48, 3)
    assert frame[:, :, 1].mean() > 180
    assert frame[:, :, 0].mean() < 80


def test_fasterliveportrait_runtime_config_update_accepts_reference_controls() -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=_png_bytes((32, 32)),
        config={"width": 32, "height": 32},
    )

    updated = service.update_runtime_config(
        session.session_id,
        {
            "flag_stitching": True,
            "flag_pasteback": True,
            "flag_relative_motion": False,
            "flag_normalize_lip": False,
            "flag_lip_retargeting": True,
            "head_only_pasteback": False,
            "cfg_scale": 1.2,
            "width": 999,
        },
    )

    assert updated == {
        "flag_stitching": True,
        "flag_pasteback": True,
        "flag_relative_motion": False,
        "flag_normalize_lip": False,
        "flag_lip_retargeting": True,
        "head_only_pasteback": False,
        "cfg_scale": 1.2,
    }
    assert session.runtime_config["flag_pasteback"] is True
    assert session.runtime_config["flag_relative_motion"] is False
    assert session.video.width == 32


def test_fasterliveportrait_video_clone_prefers_animated_crop_when_stitching_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=_png_bytes((32, 32)),
        config={"width": 32, "height": 32, "flag_stitching": False},
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        source_path = None
        run_kwargs = None

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def run(self, image, img_src, src_info, **kwargs):
            self.run_kwargs = dict(kwargs)
            if "realtime" in kwargs:
                raise TypeError("FasterLivePortraitPipeline._run() got multiple values for argument 'realtime'")
            driving_crop = np.zeros((32, 32, 3), dtype=np.uint8)
            animated_crop = np.full((32, 32, 3), (240, 20, 20), dtype=np.uint8)
            static_pasteback = np.full((32, 32, 3), (20, 20, 240), dtype=np.uint8)
            return driving_crop, animated_crop, static_pasteback, ({}, None, None)

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"pipeline": fake_pipeline})

    payload = runtime._render_model_driving_frame(session, _png_bytes((48, 48)), state)
    frame = np.asarray(Image.open(io.BytesIO(decode_jpeg_sequence(payload)[0])).convert("RGB"))

    assert fake_pipeline.run_kwargs == {"first_frame": True}
    assert frame[:, :, 0].mean() > 200
    assert frame[:, :, 2].mean() < 80
    assert state.driving_frame_index == 1


def test_fasterliveportrait_model_render_uses_realtime_run_path(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={"chunk_samples": 8000, "emit_frames_per_chunk": 1, "width": 32, "height": 32},
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None
        called_run = False

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def run_with_pkl(self, motion_info, img_src, src_info, **kwargs):
            raise AssertionError("realtime rendering should bypass run_with_pkl to avoid pasteback overhead")

        def _run(
            self,
            src_info,
            x_d_i_info,
            x_d_0_info,
            R_d_i,
            R_d_0,
            realtime,
            input_eye_ratio,
            input_lip_ratio,
            I_p_pstbk,
        ):
            assert realtime is True
            assert x_d_0_info == x_d_i_info
            assert R_d_0 == [[1]]
            self.called_run = True
            from PIL import Image

            return Image.new("RGB", (32, 32), "red"), None

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(
        runtime,
        "_load_model_bundle",
        lambda: {"joyvasa": object(), "pipeline": fake_pipeline},
    )
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[1]], "exp": [[1]]}],
    )

    payload = runtime._render_model_chunk(session, b"\0" * 16000, state)

    assert len(decode_jpeg_sequence(payload)) == 1
    assert fake_pipeline.called_run is True


def test_fasterliveportrait_applies_stitching_config_before_preparing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={
            "chunk_samples": 8000,
            "emit_frames_per_chunk": 1,
            "render_keyframes_per_chunk": 1,
            "width": 32,
            "height": 32,
            "flag_stitching": True,
        },
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((64, 96, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None
        stitching_at_prepare = None
        realtime_arg = None

        def __init__(self):
            class InferParams:
                flag_stitching = False

            class Cfg:
                infer_params = InferParams()

            self.cfg = Cfg()

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            self.stitching_at_prepare = self.cfg.infer_params.flag_stitching
            return True

        def update_cfg(self, args):
            if "flag_stitching" in args:
                self.cfg.infer_params.flag_stitching = args["flag_stitching"]

        def _run(
            self,
            src_info,
            x_d_i_info,
            x_d_0_info,
            R_d_i,
            R_d_0,
            realtime,
            input_eye_ratio,
            input_lip_ratio,
            I_p_pstbk,
        ):
            self.realtime_arg = realtime
            crop = np.zeros((32, 32, 3), dtype=np.uint8)
            pasted = np.full((64, 96, 3), 180, dtype=np.uint8)
            return crop, pasted

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": fake_pipeline})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[1]], "exp": [[1]]}],
    )

    payload = runtime._render_model_chunk(session, b"\0" * 16000, state)

    assert fake_pipeline.stitching_at_prepare is True
    assert fake_pipeline.realtime_arg is False
    assert len(decode_jpeg_sequence(payload)) == 1


def test_fasterliveportrait_render_keyframes_expand_to_emit_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={"chunk_samples": 8000, "emit_frames_per_chunk": 16, "render_keyframes_per_chunk": 4, "width": 32, "height": 32},
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None
        calls = 0

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def _run(self, *args):
            self.calls += 1
            from PIL import Image

            return Image.new("RGB", (32, 32), (self.calls * 30, 0, 0)), None

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": fake_pipeline})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[idx]], "exp": [[idx]]} for idx in range(16)],
    )

    payload = runtime._render_model_chunk(session, b"\0" * 16000, state)

    assert fake_pipeline.calls == 4
    assert len(decode_jpeg_sequence(payload)) == 16


def test_fasterliveportrait_can_disable_frame_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={
            "chunk_samples": 8000,
            "emit_frames_per_chunk": 16,
            "render_keyframes_per_chunk": 4,
            "disable_frame_interpolation": True,
            "width": 32,
            "height": 32,
        },
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None
        rendered: list[int] = []

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def _run(self, src_info, x_d_i_info, *args):
            value = int(x_d_i_info["exp"][0][0])
            self.rendered.append(value)
            from PIL import Image

            return Image.new("RGB", (32, 32), (value * 40, 0, 0)), None

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": fake_pipeline})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[idx]], "exp": [[idx]]} for idx in range(3)],
    )

    payload = runtime._render_model_chunk(session, b"\0" * 16000, state)

    assert fake_pipeline.rendered == [0, 1, 2]
    assert len(decode_jpeg_sequence(payload)) == 3


def test_fasterliveportrait_disable_interpolation_can_downsample_raw_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={
            "chunk_samples": 16000,
            "emit_frames_per_chunk": 12,
            "disable_frame_interpolation": True,
            "width": 32,
            "height": 32,
        },
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None
        rendered: list[int] = []

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def _run(self, src_info, x_d_i_info, *args):
            value = int(x_d_i_info["exp"][0][0])
            self.rendered.append(value)
            from PIL import Image

            return Image.new("RGB", (32, 32), (value % 255, 0, 0)), None

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": fake_pipeline})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[idx]], "exp": [[idx]]} for idx in range(25)],
    )

    payload = runtime._render_model_chunk(session, b"\0" * 32000, state)

    assert len(decode_jpeg_sequence(payload)) == 12
    assert fake_pipeline.rendered == [0, 2, 4, 7, 9, 11, 13, 15, 17, 20, 22, 24]


def test_fasterliveportrait_passes_audio_lip_ratios_to_retargeting(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={
            "chunk_samples": 8000,
            "emit_frames_per_chunk": 4,
            "render_keyframes_per_chunk": 4,
            "width": 32,
            "height": 32,
            "flag_lip_retargeting": True,
            "lip_retargeting_multiplier": 5.0,
            "lip_retargeting_noise_floor": 0.0,
            "lip_retargeting_min": 0.01,
            "lip_retargeting_max": 0.6,
        },
    )
    state = runtime._session_state(session)
    lip_ratios: list[float | None] = []
    update_args: dict[str, object] = {}

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            update_args.update(args)

        def _run(
            self,
            src_info,
            x_d_i_info,
            x_d_0_info,
            R_d_i,
            R_d_0,
            realtime,
            input_eye_ratio,
            input_lip_ratio,
            I_p_pstbk,
        ):
            lip_ratios.append(input_lip_ratio)
            from PIL import Image

            return Image.new("RGB", (32, 32), "red"), None

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": fake_pipeline})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[idx]], "exp": [[idx]]} for idx in range(4)],
    )
    low = np.zeros(2000, dtype=np.int16)
    high = np.full(2000, 12000, dtype=np.int16)
    audio = np.concatenate([low, high, high, low]).astype(np.int16).tobytes()

    payload = runtime._render_model_chunk(session, audio, state)

    assert len(decode_jpeg_sequence(payload)) == 4
    assert update_args["flag_lip_retargeting"] is True
    assert lip_ratios[0] < lip_ratios[1]
    assert lip_ratios[2] > lip_ratios[3]



def test_fasterliveportrait_forces_eager_attention_for_joyvasa_audio_encoders() -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeAudioModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: object) -> object:
            calls.append((source, kwargs))
            return object()

    runtime._force_eager_attention_for_audio_encoder(FakeAudioModel, "hubert")
    FakeAudioModel.from_pretrained("/tmp/hubert")
    FakeAudioModel.from_pretrained("/tmp/hubert-explicit", attn_implementation="eager")

    assert calls == [
        ("/tmp/hubert", {"attn_implementation": "eager"}),
        ("/tmp/hubert-explicit", {"attn_implementation": "eager"}),
    ]
    assert FakeAudioModel.from_pretrained.__func__._omnirt_forces_eager_attention is True


def test_fasterliveportrait_trt_config_requires_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    import omnirt.models.fasterliveportrait.runtime as flp_runtime

    monkeypatch.setattr(
        flp_runtime.importlib.util,
        "find_spec",
        lambda name: None if name == "tensorrt" else object(),
    )

    with pytest.raises(FasterLivePortraitRuntimeError, match="TensorRT is required"):
        runtime._validate_runtime_dependencies(Path("configs/trt_infer.yaml"))


def test_fasterliveportrait_trt_config_requires_grid_sample_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(
        checkpoints_dir="/tmp/missing-fasterliveportrait-checkpoints",
        load_models=False,
    )
    import omnirt.models.fasterliveportrait.runtime as flp_runtime

    monkeypatch.setattr(flp_runtime.importlib.util, "find_spec", lambda name: object())

    with pytest.raises(FasterLivePortraitRuntimeError, match="GridSample3D TensorRT plugin"):
        runtime._validate_runtime_dependencies(Path("configs/trt_infer.yaml"))


def test_fasterliveportrait_resamples_sparse_motion_before_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={
            "chunk_samples": 8000,
            "emit_frames_per_chunk": 5,
            "render_keyframes_per_chunk": 5,
            "width": 32,
            "height": 32,
        },
    )
    state = runtime._session_state(session)
    rendered_exp: list[float] = []

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def _run(self, src_info, x_d_i_info, *args):
            value = float(x_d_i_info["exp"][0, 0, 0])
            rendered_exp.append(value)
            from PIL import Image

            return Image.new("RGB", (32, 32), (int(value * 2), 0, 0)), None

    fake_pipeline = FakePipeline()
    start = np.zeros((1, 21, 3), dtype=np.float32)
    end = np.full((1, 21, 3), 100, dtype=np.float32)
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": fake_pipeline})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [
            {
                "R": np.zeros((1, 3, 3), dtype=np.float32),
                "exp": start,
                "scale": np.ones((1, 1), dtype=np.float32),
                "t": np.zeros((1, 3), dtype=np.float32),
                "pitch": np.zeros((1, 1), dtype=np.float32),
                "yaw": np.zeros((1, 1), dtype=np.float32),
                "roll": np.zeros((1, 1), dtype=np.float32),
            },
            {
                "R": np.ones((1, 3, 3), dtype=np.float32),
                "exp": end,
                "scale": np.ones((1, 1), dtype=np.float32),
                "t": np.ones((1, 3), dtype=np.float32),
                "pitch": np.ones((1, 1), dtype=np.float32),
                "yaw": np.ones((1, 1), dtype=np.float32),
                "roll": np.ones((1, 1), dtype=np.float32),
            },
        ],
    )

    payload = runtime._render_model_chunk(session, b"\0" * 16000, state)

    assert len(decode_jpeg_sequence(payload)) == 5
    assert rendered_exp == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_fasterliveportrait_chunk_observability_logs_and_prints_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={
            "chunk_samples": 8000,
            "emit_frames_per_chunk": 2,
            "render_keyframes_per_chunk": 2,
            "width": 32,
            "height": 32,
        },
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def _run(self, *args):
            from PIL import Image

            return Image.new("RGB", (32, 32), "red"), None

    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": FakePipeline()})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[idx]], "exp": [[idx]]} for idx in range(2)],
    )
    logged_messages: list[str] = []

    def capture_log(message: str) -> None:
        logged_messages.append(message)

    monkeypatch.setattr(flp_runtime.log, "info", capture_log)

    runtime._render_model_chunk(session, b"\0" * 16000, state)

    stdout_text = capsys.readouterr().out
    assert logged_messages
    for text in (logged_messages[-1], stdout_text):
        assert "FasterLivePortrait chunk rendered:" in text
        assert "frames=2" in text
        assert "motion_ms=" in text
        assert "render_ms=" in text
        assert "encode_ms=" in text
        assert "payload_kb=" in text
        assert "gpu_mem_mb=" in text


def test_fasterliveportrait_interpolates_repeated_keyframe_jpegs() -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    from PIL import Image
    import io

    def jpeg(red: int) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (red, 0, 0)).save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    frames = runtime._expand_keyframes([jpeg(0), jpeg(0), jpeg(120), jpeg(120)], 5)
    reds = [
        int(np.asarray(Image.open(io.BytesIO(frame)).convert("RGB"))[0, 0, 0])
        for frame in frames
    ]

    assert len(frames) == 5
    assert len(set(frames)) == 5
    assert reds[0] < reds[1] < reds[2] < reds[3] < reds[4]


def test_fasterliveportrait_prefers_dynamic_crop_when_org_is_static(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={"chunk_samples": 8000, "emit_frames_per_chunk": 2, "render_keyframes_per_chunk": 2, "width": 32, "height": 32},
    )
    state = runtime._session_state(session)

    class FakePipeline:
        src_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]
        src_infos = ["src"]
        device = "cpu"
        source_path = None
        R_d_0 = None
        x_d_0_info = None
        calls = 0

        def init_vars(self):
            pass

        def prepare_source(self, source_path, **kwargs):
            self.source_path = source_path
            return True

        def update_cfg(self, args):
            pass

        def _run(self, *args):
            self.calls += 1
            crop = np.full((32, 32, 3), self.calls * 80, dtype=np.uint8)
            org = np.zeros((32, 32, 3), dtype=np.uint8)
            return crop, org

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runtime, "_load_model_bundle", lambda: {"joyvasa": object(), "pipeline": fake_pipeline})
    monkeypatch.setattr(
        runtime,
        "_generate_motion_infos",
        lambda joyvasa, pcm_s16le, state, session: [{"R": [[idx]], "exp": [[idx]]} for idx in range(2)],
    )

    payload = runtime._render_model_chunk(session, b"\0" * 16000, state)
    frames = decode_jpeg_sequence(payload)

    assert len(frames) == 2
    assert frames[0] != frames[1]


def test_fasterliveportrait_head_only_pasteback_keeps_body_from_reference() -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[40:64, 18:46] = (20, 180, 20)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=_png_bytes(base),
        config={"width": 64, "height": 64, "flag_stitching": True},
    )
    state = runtime._session_state(session)
    generated = base.copy()
    generated[12:30, 24:40] = (220, 30, 30)
    generated[40:64, 18:46] = (20, 20, 220)

    composed = runtime._compose_head_on_reference(generated, session, state)

    assert np.mean(composed[14:28, 26:38, 0]) > 120
    assert np.array_equal(composed[44:60, 22:42], base[44:60, 22:42])



def test_fasterliveportrait_pose_multiplier_only_scales_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={"pose_motion_multiplier": 0.25},
    )
    motion_info = {
        "R": np.ones((1, 3, 3), dtype=np.float32),
        "exp": np.full((1, 21, 3), 7.0, dtype=np.float32),
        "scale": np.array([[2.0]], dtype=np.float32),
        "t": np.array([[4.0, 8.0, 12.0]], dtype=np.float32),
        "pitch": np.array([[1.0]], dtype=np.float32),
        "yaw": np.array([[2.0]], dtype=np.float32),
        "roll": np.array([[3.0]], dtype=np.float32),
    }

    fake_utils = types.ModuleType("src.utils.utils")
    fake_utils.get_rotation_matrix = lambda pitch, yaw, roll: np.array(
        [
            [float(pitch[0, 0]), float(yaw[0, 0]), float(roll[0, 0])],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    fake_src = types.ModuleType("src")
    fake_src_utils = types.ModuleType("src.utils")
    fake_src_utils.utils = fake_utils
    monkeypatch.setitem(sys.modules, "src", fake_src)
    monkeypatch.setitem(sys.modules, "src.utils", fake_src_utils)
    monkeypatch.setitem(sys.modules, "src.utils.utils", fake_utils)

    scaled = runtime._apply_pose_motion_multiplier(motion_info, session)

    assert np.array_equal(scaled["exp"], motion_info["exp"])
    assert scaled["pitch"][0, 0] == pytest.approx(0.25)
    assert scaled["yaw"][0, 0] == pytest.approx(0.5)
    assert scaled["roll"][0, 0] == pytest.approx(0.75)
    assert scaled["scale"][0, 0] == pytest.approx(2.0)
    assert scaled["t"].tolist() == [[4.0, 8.0, 12.0]]
    assert scaled["R"][0, 0].tolist() == [0.25, 0.5, 0.75]
    assert motion_info["pitch"][0, 0] == pytest.approx(1.0)

def test_fasterliveportrait_cfg_scale_zero_disables_joyvasa_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FasterLivePortraitRealtimeRuntime(load_models=False)
    service = RealtimeAvatarService(runtime=runtime)
    session = service.create_session(
        model=FASTLIVEPORTRAIT_MODEL_ID,
        backend="cpu-stub",
        image_bytes=b"fake-image-bytes",
        config={"chunk_samples": 8000, "cfg_scale": 0},
    )
    state = runtime._session_state(session)

    class FakeMotionGenerator:
        def sample(self, audio_in, *args, **kwargs):
            assert kwargs["cfg_cond"] == []
            raise RuntimeError("stop after cfg assertion")

    class FakeJoyVASA:
        device = "cpu"
        dtype = __import__("torch").float32
        n_audio_samples = 64000
        fps = 25
        n_motions = 100
        audio_unit = 640
        pad_mode = "zero"
        use_indicator = False
        cfg_scale = 3.5
        cfg_cond = ["audio"]
        cfg_mode = "incremental"
        motion_generator = FakeMotionGenerator()

    with pytest.raises(RuntimeError, match="stop after cfg assertion"):
        runtime._generate_motion_infos(FakeJoyVASA(), b"\0\0" * 64000, state, session)


def test_fasterliveportrait_compatible_ws_init_generate_and_close() -> None:
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=FasterLivePortraitRealtimeRuntime(load_models=False)
    )
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/fasterliveportrait") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "chunk_samples": 8000,
                "emit_frames_per_chunk": 12,
                "head_motion_multiplier": 1.0,
                "pose_motion_multiplier": 0.4,
                "animation_region": "lip",
                "expression_multiplier": 1.1,
                "mouth_open_multiplier": 2.0,
                "mouth_corner_multiplier": 1.2,
                "cheek_jaw_multiplier": 0.9,
                "driving_multiplier": 1.1,
                "cfg_scale": 3.5,
                "lookahead_ms": 240,
            }
        )
        init = ws.receive_json()
        assert init["type"] == "init_ok"
        assert init["model"] == FASTLIVEPORTRAIT_MODEL_ID
        assert init["slice_len"] == 12
        assert init["chunk_samples"] == 8000
        session_id = next(iter(app.state.realtime_avatar_service._sessions))
        session = app.state.realtime_avatar_service._sessions[session_id]
        assert session.runtime_config["pose_motion_multiplier"] == 0.4
        assert session.runtime_config["animation_region"] == "lip"
        assert session.runtime_config["mouth_open_multiplier"] == 2.0
        assert session.runtime_config["mouth_corner_multiplier"] == 1.2
        assert session.runtime_config["cheek_jaw_multiplier"] == 0.9
        assert session.runtime_config["driving_multiplier"] == 1.1

        ws.send_bytes(_audio_payload(8000))
        video = ws.receive_bytes()
        assert len(decode_jpeg_sequence(video)) == 12

        ws.send_json({"type": "close"})
        assert ws.receive_json()["type"] == "close_ok"


def test_flashtalk_compatible_ws_offloads_audio_push(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    calls = 0
    real_to_thread = avatar_routes.asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        nonlocal calls
        calls += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(avatar_routes.asyncio, "to_thread", tracking_to_thread)
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/avatar/flashtalk") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64()})
        init = ws.receive_json()
        ws.send_bytes(_audio_payload(init["slice_len"] * 16000 // init["fps"]))
        video = ws.receive_bytes()

    assert video[:4] == MAGIC_VIDEO
    assert calls == 1


def test_flashtalk_compatible_ws_root_alias_for_opentalking_default() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64()})
        assert ws.receive_json()["type"] == "init_ok"


def test_audio2video_models_reports_wav2lip_unavailable_by_default() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == []
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["connected"] is False
    assert statuses["flashtalk"]["reason"] == "fallback_runtime"
    assert statuses["wav2lip"]["connected"] is False
    assert statuses["quicktalk"]["connected"] is False
    assert statuses["quicktalk"]["reason"] == "runtime_not_enabled"


def test_audio2video_models_reports_resident_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_REALTIME_AVATAR_RUNTIME", "resident")
    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["connected"] is True
    assert statuses["flashtalk"]["reason"] == "resident_runtime"


def test_avatar_models_alias_reports_wav2lip_unavailable_by_default() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/avatar/models")

    assert response.status_code == 200
    assert response.json()["models"] == []


def test_audio2video_models_reports_proxy_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    async def fake_reachable(_url: str) -> bool:
        return True

    monkeypatch.setattr(avatar_routes, "_is_ws_url_reachable", fake_reachable)
    client = TestClient(create_app(default_backend="cpu-stub"))
    client.app.state.avatar_model_ws_urls = {
        "flashtalk": "ws://127.0.0.1:8765",
        "wav2lip": "ws://127.0.0.1:8767",
        "quicktalk": "ws://127.0.0.1:8768",
    }

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["flashtalk", "wav2lip", "quicktalk"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["reason"] == "proxy"
    assert statuses["wav2lip"]["connected"] is True
    assert statuses["quicktalk"]["connected"] is True
    assert statuses["quicktalk"]["reason"] == "proxy"


def test_audio2video_models_reports_musetalk_proxy_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    async def fake_reachable(_url: str) -> bool:
        return True

    monkeypatch.setattr(avatar_routes, "_is_ws_url_reachable", fake_reachable)
    client = TestClient(create_app(default_backend="cpu-stub"))
    client.app.state.avatar_model_ws_urls = {
        "musetalk": "ws://127.0.0.1:8766",
    }

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["musetalk"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["musetalk"]["connected"] is True
    assert statuses["musetalk"]["reason"] == "proxy"
    assert statuses["flashtalk"]["connected"] is False
    assert statuses["wav2lip"]["connected"] is False


def test_audio2video_models_reads_proxy_targets_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    async def fake_reachable(_url: str) -> bool:
        return True

    monkeypatch.setenv("OMNIRT_AVATAR_FLASHTALK_WS_URL", "ws://127.0.0.1:8765")
    monkeypatch.setenv("OMNIRT_AVATAR_WAV2LIP_WS_URL", "ws://127.0.0.1:8767")
    monkeypatch.setenv("OMNIRT_AVATAR_QUICKTALK_WS_URL", "ws://127.0.0.1:8768")
    monkeypatch.setattr(avatar_routes, "_is_ws_url_reachable", fake_reachable)

    client = TestClient(create_app(default_backend="cpu-stub"))
    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["flashtalk", "wav2lip", "quicktalk"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["reason"] == "proxy"
    assert statuses["wav2lip"]["reason"] == "proxy"
    assert statuses["quicktalk"]["reason"] == "proxy"


def test_audio2video_models_reports_quicktalk_runtime() -> None:
    class FakeRouter:
        runtime_kind = "router"
        wav2lip = None
        quicktalk = object()

    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=FakeRouter())
    client = TestClient(app)

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert "quicktalk" in payload["models"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["quicktalk"]["connected"] is True
    assert statuses["quicktalk"]["reason"] == "quicktalk_runtime"


def test_audio2video_models_reads_musetalk_proxy_target_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    async def fake_reachable(_url: str) -> bool:
        return True

    monkeypatch.setenv("OMNIRT_AVATAR_MUSETALK_WS_URL", "ws://127.0.0.1:8766")
    monkeypatch.setattr(avatar_routes, "_is_ws_url_reachable", fake_reachable)

    client = TestClient(create_app(default_backend="cpu-stub"))
    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["musetalk"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["musetalk"]["connected"] is True
    assert statuses["musetalk"]["reason"] == "proxy"


def test_flashtalk_compatible_ws_errors() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/flashtalk") as ws:
        ws.send_json({"type": "init"})
        missing = ws.receive_json()
        assert missing["type"] == "error"
        assert missing["code"] == "missing_image"

        ws.send_json({"type": "init", "ref_image": "not-base64"})
        bad_b64 = ws.receive_json()
        assert bad_b64["type"] == "error"
        assert bad_b64["code"] == "bad_image_base64"

        ws.send_json({"type": "init", "ref_image": _image_b64()})
        init = ws.receive_json()
        assert init["type"] == "init_ok"

        ws.send_bytes(b"NOPE")
        bad_magic = ws.receive_json()
        assert bad_magic["type"] == "error"
        assert bad_magic["code"] == "bad_audio_magic"

        ws.send_bytes(MAGIC_AUDIO + b"\0")
        bad_chunk = ws.receive_json()
        assert bad_chunk["type"] == "error"
        assert bad_chunk["code"] == "bad_audio_chunk"


def test_flashtalk_compatible_ws_reports_runtime_errors() -> None:
    class FailingRuntime:
        def render_chunk(self, session, pcm_s16le):
            del session, pcm_s16le
            raise RuntimeError("model failed")

    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=FailingRuntime())
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/flashtalk") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64()})
        init = ws.receive_json()
        ws.send_bytes(_audio_payload(init["slice_len"] * 16000 // init["fps"]))
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "runtime_error"
    assert "model failed" in error["message"]


def test_musetalk_compatible_ws_reports_proxy_not_configured() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/musetalk") as ws:
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "musetalk_proxy_not_configured"
    assert "OMNIRT_AVATAR_MUSETALK_WS_URL" in error["message"]


def test_musetalk_compatible_ws_uses_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    calls: list[tuple[object, str]] = []

    async def fake_proxy(websocket, target_url: str) -> None:
        calls.append((websocket, target_url))
        await websocket.accept()
        await websocket.send_json({"type": "proxied", "target": target_url})
        await websocket.close()

    monkeypatch.setattr(avatar_routes, "_proxy_websocket", fake_proxy)
    client = TestClient(create_app(default_backend="cpu-stub"))
    client.app.state.avatar_model_ws_urls = {
        "musetalk": "ws://127.0.0.1:8766",
    }

    with client.websocket_connect("/v1/audio2video/musetalk") as ws:
        proxied = ws.receive_json()

    assert proxied["type"] == "proxied"
    assert proxied["target"] == "ws://127.0.0.1:8766"
    assert len(calls) == 1


def test_native_realtime_avatar_ws_flow() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/avatar/realtime") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "session.create",
                    "model": "soulx-flashtalk-14b",
                    "backend": "cpu-stub",
                    "inputs": {"image_b64": _image_b64(), "prompt": "talk"},
                    "config": {"chunk_samples": 16, "width": 32, "height": 32},
                }
)
        )
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session_id"].startswith("avt_")
        assert created["trace_id"].startswith("trace_")
        assert created["audio"]["chunk_samples"] == 16
        assert created["video"]["width"] == 32

        ws.send_bytes(_audio_payload(16))
        metrics = ws.receive_json()
        assert metrics["type"] == "metrics"
        assert metrics["chunk_index"] == 1
        video = ws.receive_bytes()
        assert video[:4] == MAGIC_VIDEO

        ws.send_json({"type": "session.cancel"})
        assert ws.receive_json()["type"] == "session.cancelled"

        ws.send_json({"type": "session.close"})
        assert ws.receive_json()["type"] == "session.closed"


def test_realtime_avatar_cancel_releases_runtime_session_state() -> None:
    closed: list[str] = []

    class RuntimeWithState(FakeRealtimeAvatarRuntime):
        def close_session(self, session_id: str) -> None:
            closed.append(session_id)

    service = RealtimeAvatarService(runtime=RuntimeWithState())
    session = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((32, 32)),
        config={"chunk_samples": 16, "width": 32, "height": 32},
    )

    service.cancel_session(session.session_id)

    assert closed == [session.session_id]
    assert service._sessions[session.session_id].cancelled is True


def test_wav2lip_init_accepts_postprocess_mode_and_metadata() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    metadata = {
        "source_image_hash": "abc123",
        "animation": {
            "mouth_center": [0.5, 0.56],
            "mouth_rx": 0.06,
            "mouth_ry": 0.02,
            "outer_lip": [[0.45, 0.55], [0.50, 0.53], [0.55, 0.55], [0.50, 0.58]],
            "inner_mouth": [[0.47, 0.55], [0.53, 0.55], [0.50, 0.57]],
        },
    }

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "wav2lip_postprocess_mode": "opentalking_improved",
                "mouth_metadata": metadata,
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "wav2lip"
    assert init["wav2lip_postprocess_mode"] == "opentalking_improved"


def test_wav2lip_init_accepts_frame_reference_dir(tmp_path: Path) -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    client.app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "reference_mode": "frames",
                "ref_frame_dir": str(frame_dir),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "wav2lip"
    assert init["reference_mode"] == "frames"
    assert "ref_frame_dir" not in init


def test_quicktalk_compatible_ws_accepts_template_video(tmp_path: Path) -> None:
    template = tmp_path / "template.mp4"
    template.write_bytes(b"video")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "template_mode": "video",
                "template_video": str(template),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "quicktalk"
    assert init["template_mode"] == "video"
    assert "template_video" not in init


def test_quicktalk_compatible_ws_accepts_template_frame_dir(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "template_mode": "frames",
                "template_frame_dir": str(frame_dir),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "quicktalk"
    assert init["template_mode"] == "frames"


def test_quicktalk_template_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    template = outside / "template.mp4"
    template.write_bytes(b"video")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[allowed])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "template_mode": "video",
                "template_video": str(template),
            }
        )
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_template_video"


def test_wav2lip_video_dimensions_respect_max_long_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_WAV2LIP_MAX_LONG_EDGE", "768")
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "width": 830,
                "height": 1108,
                "fps": 30,
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["width"] == 574
    assert init["height"] == 768


def test_quicktalk_video_dimensions_default_to_900_long_edge() -> None:
    service = RealtimeAvatarService()
    session = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((1600, 1200)),
        config={"width": 1600, "height": 1200},
    )

    assert session.video.width == 900
    assert session.video.height == 674
    assert session.video.fps == 25


def test_quicktalk_video_dimensions_respect_max_long_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_QUICKTALK_MAX_LONG_EDGE", "512")
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "width": 830,
                "height": 1108,
                "fps": 30,
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["width"] == 384
    assert init["height"] == 512
    assert init["fps"] == 25


def test_quicktalk_defaults_to_low_latency_streaming_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_QUICKTALK_REALTIME_CHUNK_MS", "160")
    monkeypatch.delenv("OMNIRT_QUICKTALK_STREAMING_LOOKAHEAD_CHUNKS", raising=False)
    service = RealtimeAvatarService()

    session = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"width": 64, "height": 64},
    )

    assert session.video.fps == 25
    assert session.video.slice_len == 4
    assert session.audio.chunk_samples == 2560
    assert session.lookahead_chunks == 1


def test_quicktalk_can_disable_streaming_lookahead_for_latency_experiments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIRT_QUICKTALK_REALTIME_CHUNK_MS", "160")
    monkeypatch.setenv("OMNIRT_QUICKTALK_STREAMING_LOOKAHEAD_CHUNKS", "0")
    service = RealtimeAvatarService()

    session = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"width": 64, "height": 64},
    )

    assert session.video.slice_len == 4
    assert session.audio.chunk_samples == 2560
    assert session.lookahead_chunks == 0


def test_quicktalk_explicit_slice_len_keeps_legacy_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_QUICKTALK_REALTIME_CHUNK_MS", "160")
    service = RealtimeAvatarService()

    session = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"width": 64, "height": 64, "slice_len": 28},
    )

    assert session.video.slice_len == 28
    assert session.audio.chunk_samples == 17920


def test_quicktalk_init_accepts_asset_face_cache_path(tmp_path: Path) -> None:
    cache_path = tmp_path / "quicktalk" / "face_cache_v3_900.npz"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"cache")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "quicktalk_face_cache": str(cache_path),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "quicktalk"


def test_quicktalk_face_cache_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    cache_path = outside / "face_cache.npz"
    cache_path.write_bytes(b"cache")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[allowed])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "quicktalk_face_cache": str(cache_path),
            }
        )
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_quicktalk_face_cache"


def test_quicktalk_static_template_video_uses_session_dimensions(tmp_path: Path) -> None:
    from omnirt.models.quicktalk.runtime import QuickTalkRealtimeRuntime

    session = RealtimeAvatarService().create_session(
        model="quicktalk",
        image_bytes=_png_bytes((2048, 2048)),
        config={"width": 512, "height": 512},
    )
    runtime = QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )

    template = runtime._template_video_for(session)

    import cv2

    cap = cv2.VideoCapture(str(template))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 512
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 512
    finally:
        cap.release()


def test_quicktalk_runtime_uses_streaming_pcm_feature_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    class FakeState:
        def __init__(self) -> None:
            self.calls: list[int] = []

    class FakeWorker:
        def make_state(self) -> FakeState:
            return FakeState()

        def prepare_streaming_pcm_features(self, pcm, sample_rate, *, state):
            state.calls.append(int(pcm.size))
            return [np.zeros((10, 1024), dtype=np.float32)], 0.0

        def generate_frames_from_reps(self, reps, state=None):
            yield np.zeros((16, 16, 3), dtype=np.uint8)

    runtime = quicktalk_runtime.QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )
    monkeypatch.setattr(runtime, "_worker_for", lambda session: FakeWorker())
    session = RealtimeAvatarService().create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"chunk_samples": 2560, "slice_len": 4, "width": 64, "height": 64},
    )

    payload = b"\0\0" * session.audio.chunk_samples
    runtime.render_chunk(session, payload)
    runtime.render_chunk(session, payload)

    state = runtime._states[session.session_id]
    assert state.calls == [2560, 2560]


def test_quicktalk_streaming_features_delay_until_lookahead_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk.runtime_worker import RealtimeV3SessionState, RealtimeV3Worker

    monkeypatch.setenv("OMNIRT_QUICKTALK_STREAMING_CONTEXT_MS", "100")
    monkeypatch.setenv("OMNIRT_QUICKTALK_STREAMING_LOOKAHEAD_CHUNKS", "1")

    worker = object.__new__(RealtimeV3Worker)
    worker.fps = 25
    seen: list[tuple[int, int]] = []

    def fake_prepare_pcm_features(pcm, sample_rate):
        arr = np.asarray(pcm, dtype=np.int16).reshape(-1)
        seen.append((int(arr[0]) if arr.size else -1, int(arr[-1]) if arr.size else -1))
        frames = max(1, int(arr.size / sample_rate * worker.fps))
        return [np.full((10, 1024), idx, dtype=np.float32) for idx in range(frames)], 0.0

    worker.prepare_pcm_features = fake_prepare_pcm_features  # type: ignore[method-assign]
    state = RealtimeV3SessionState()
    first = np.full(1600, 1, dtype=np.int16)
    second = np.full(1600, 2, dtype=np.int16)
    third = np.full(1600, 3, dtype=np.int16)

    reps0, _ = worker.prepare_streaming_pcm_features(first, 16000, state=state)
    reps1, _ = worker.prepare_streaming_pcm_features(second, 16000, state=state)
    reps2, _ = worker.prepare_streaming_pcm_features(third, 16000, state=state)
    flush, _ = worker.flush_streaming_pcm_features(16000, state=state)

    assert reps0 == []
    assert len(reps1) == 2
    assert len(reps2) == 2
    assert len(flush) == 2
    assert seen == [(1, 2), (1, 3), (2, 0)]
    assert state.pcm_history.size == 1600
    assert state.pending_pcm.size == 0


def test_quicktalk_streaming_features_match_full_audio_frame_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.models.quicktalk.runtime_worker import RealtimeV3SessionState, RealtimeV3Worker

    monkeypatch.setenv("OMNIRT_QUICKTALK_STREAMING_CONTEXT_MS", "500")
    monkeypatch.setenv("OMNIRT_QUICKTALK_STREAMING_LOOKAHEAD_CHUNKS", "1")

    worker = object.__new__(RealtimeV3Worker)
    worker.fps = 25

    def fake_prepare_pcm_features(pcm, sample_rate):
        arr = np.asarray(pcm, dtype=np.int16).reshape(-1)
        frames = max(1, int(arr.size / sample_rate * worker.fps))
        # Encode the frame's first sample into the rep so the test can verify
        # that streaming returns the same temporal frame windows as full audio.
        reps = []
        for frame_idx in range(frames):
            start = int(frame_idx * sample_rate / worker.fps)
            reps.append(np.full((10, 1024), int(arr[min(start, arr.size - 1)]), dtype=np.float32))
        return reps, 0.0

    worker.prepare_pcm_features = fake_prepare_pcm_features  # type: ignore[method-assign]
    state = RealtimeV3SessionState()
    chunks = [
        np.full(2560, value, dtype=np.int16)
        for value in (10, 20, 30, 40)
    ]

    stream_reps: list[np.ndarray] = []
    for chunk in chunks:
        reps, _ = worker.prepare_streaming_pcm_features(chunk, 16000, state=state)
        stream_reps.extend(reps)
    flush, _ = worker.flush_streaming_pcm_features(16000, state=state)
    stream_reps.extend(flush)

    full_reps, _ = worker.prepare_pcm_features(np.concatenate(chunks), 16000)

    assert [int(rep[0, 0]) for rep in stream_reps[: len(full_reps)]] == [
        int(rep[0, 0]) for rep in full_reps
    ]


def test_quicktalk_runtime_preload_warms_streaming_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIRT_QUICKTALK_STREAMING_LOOKAHEAD_CHUNKS", raising=False)
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    class FakeState:
        pass

    class FakeWorker:
        restore_contexts = [object()]

        def __init__(self) -> None:
            self.state = FakeState()
            self.streaming_calls: list[int] = []
            self.generate_states: list[object] = []

        def make_state(self) -> FakeState:
            return self.state

        def prepare_streaming_pcm_features(self, pcm, sample_rate, *, state):
            self.streaming_calls.append(int(np.asarray(pcm).size))
            assert sample_rate == 16000
            assert state is self.state
            return [np.zeros((10, 1024), dtype=np.float32)], 0.0

        def generate_frames_from_reps(self, reps, state=None):
            self.generate_states.append(state)
            yield np.zeros((16, 16, 3), dtype=np.uint8)

    worker = FakeWorker()
    runtime = quicktalk_runtime.QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )
    monkeypatch.setattr(
        runtime,
        "_worker_for_with_cache_status",
        lambda session: (worker, False),
    )
    session = RealtimeAvatarService().create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"chunk_samples": 2560, "slice_len": 4, "width": 64, "height": 64},
    )

    result = runtime.preload_reference(session)

    assert worker.streaming_calls == [2560, 2560]
    assert worker.generate_states == [worker.state, worker.state]
    assert result["warmup_chunks"] == 2
    assert result["warmup_frames"] == 1


def test_quicktalk_runtime_passes_asset_face_cache_to_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    captured: dict[str, object] = {}

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        quicktalk_runtime.QuickTalkRealtimeRuntime,
        "_worker_class",
        staticmethod(lambda: FakeWorker),
    )
    cache = tmp_path / "quicktalk" / "face_cache_v3_900.npz"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")
    session = RealtimeAvatarService(allowed_frame_roots=[tmp_path]).create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"quicktalk_face_cache": str(cache)},
    )
    runtime = quicktalk_runtime.QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )
    monkeypatch.setattr(
        runtime,
        "_template_video_for",
        lambda session: Path(session.template_video or ""),
    )

    runtime._worker_for(session)

    assert captured["face_cache_file"] == tmp_path / "quicktalk" / "face_cache_v3_900.npz"


def test_quicktalk_preload_endpoint_uses_runtime_cache(tmp_path: Path) -> None:
    class FakeQuickTalkRuntime:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def preload_reference(self, session):
            self.calls.append(session)
            return {
                "type": "preload_result",
                "frames": 25,
                "elapsed_ms": 12.5,
                "cache_hit": len(self.calls) > 1,
            }

    template = tmp_path / "quicktalk" / "template_512x512.mp4"
    cache = tmp_path / "quicktalk" / "face_cache_v3_512x512.npz"
    template.parent.mkdir()
    template.write_bytes(b"template")
    cache.write_bytes(b"cache")

    quicktalk_runtime = FakeQuickTalkRuntime()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=AvatarRuntimeRouter(
            fallback=FakeRealtimeAvatarRuntime(),
            quicktalk=quicktalk_runtime,
        ),
        allowed_frame_roots=[tmp_path],
    )
    client = TestClient(app)
    payload = {
        "template_mode": "video",
        "template_video": str(template),
        "quicktalk_face_cache": str(cache),
        "width": 512,
        "height": 512,
        "fps": 25,
    }

    first = client.post("/v1/audio2video/quicktalk/preload", json=payload)
    second = client.post("/v1/avatar/quicktalk/preload", json=payload)

    assert first.status_code == 200
    assert first.json()["cache_hit"] is False
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert len(quicktalk_runtime.calls) == 2
    assert quicktalk_runtime.calls[0].model == "quicktalk"
    assert quicktalk_runtime.calls[0].template_mode == "video"
    assert quicktalk_runtime.calls[0].template_video == str(template)
    assert quicktalk_runtime.calls[0].quicktalk_face_cache == str(cache)


def test_quicktalk_ws_init_preloads_runtime(tmp_path: Path) -> None:
    class FakeQuickTalkRuntime:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def preload_reference(self, session):
            self.calls.append(session)
            return {
                "type": "preload_result",
                "frames": 25,
                "elapsed_ms": 10.0,
                "cache_hit": False,
            }

    quicktalk_runtime = FakeQuickTalkRuntime()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=AvatarRuntimeRouter(
            fallback=FakeRealtimeAvatarRuntime(),
            quicktalk=quicktalk_runtime,
        ),
        allowed_frame_roots=[tmp_path],
    )
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "width": 512,
                "height": 512,
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "quicktalk"
    assert init["preload"]["type"] == "preload_result"
    assert init["preload"]["cache_hit"] is False
    assert len(quicktalk_runtime.calls) == 1


def test_quicktalk_startup_preload_uses_configured_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    calls: list[object] = []

    class FakeQuickTalkRuntime:
        def preload_reference(self, session):
            calls.append(session)
            return {
                "type": "preload_result",
                "frames": 25,
                "elapsed_ms": 1.0,
                "cache_hit": False,
            }

    template = tmp_path / "template.mp4"
    template.write_bytes(b"template")
    monkeypatch.setenv("OMNIRT_QUICKTALK_RUNTIME", "1")
    monkeypatch.setenv("OMNIRT_ALLOWED_FRAME_ROOTS", str(tmp_path))
    monkeypatch.setenv("OMNIRT_QUICKTALK_PRELOAD_TEMPLATE_VIDEO", str(template))
    monkeypatch.setenv("OMNIRT_QUICKTALK_PRELOAD_WIDTH", "512")
    monkeypatch.setenv("OMNIRT_QUICKTALK_PRELOAD_HEIGHT", "512")
    monkeypatch.setattr(quicktalk_runtime, "QuickTalkRealtimeRuntime", FakeQuickTalkRuntime)

    create_app(default_backend="cpu-stub")

    assert len(calls) == 1
    assert calls[0].model == "quicktalk"
    assert calls[0].template_mode == "video"
    assert calls[0].template_video == str(template.resolve())
    assert calls[0].video.width == 512
    assert calls[0].video.height == 512


def test_quicktalk_runtime_evicts_old_workers_when_cache_limit_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setenv("OMNIRT_QUICKTALK_WORKER_CACHE_MAX", "1")
    closed: list[str] = []

    class FakeWorker:
        def __init__(self, *, template_video: Path, **_: object) -> None:
            self.template_video = template_video
            self.restore_contexts = [object()]

        def make_state(self) -> object:
            return object()

        def prepare_streaming_pcm_features(self, pcm, sample_rate, *, state):
            return [np.zeros((10, 1024), dtype=np.float32)], 0.0

        def generate_frames_from_reps(self, reps, state=None):
            yield np.zeros((16, 16, 3), dtype=np.uint8)

        def close(self) -> None:
            closed.append(self.template_video.name)

    monkeypatch.setattr(
        quicktalk_runtime.QuickTalkRealtimeRuntime,
        "_worker_class",
        staticmethod(lambda: FakeWorker),
    )
    first_template = tmp_path / "first.mp4"
    second_template = tmp_path / "second.mp4"
    first_template.write_bytes(b"first")
    second_template.write_bytes(b"second")
    service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    runtime = quicktalk_runtime.QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )
    monkeypatch.setattr(
        runtime,
        "_template_video_for",
        lambda session: Path(session.template_video or ""),
    )
    first = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"template_mode": "video", "template_video": str(first_template)},
    )
    second = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"template_mode": "video", "template_video": str(second_template)},
    )

    first_result = runtime.preload_reference(first)
    second_result = runtime.preload_reference(second)

    assert first_result["cache_hit"] is False
    assert second_result["cache_hit"] is False
    assert closed == ["first.mp4"]
    assert len(runtime._workers) == 1


def test_quicktalk_runtime_preload_reports_worker_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setenv("OMNIRT_QUICKTALK_WORKER_CACHE_MAX", "2")
    build_count = 0

    class FakeWorker:
        restore_contexts = [object()]

        def __init__(self, **_: object) -> None:
            nonlocal build_count
            build_count += 1

        def make_state(self) -> object:
            return object()

        def prepare_streaming_pcm_features(self, pcm, sample_rate, *, state):
            return [np.zeros((10, 1024), dtype=np.float32)], 0.0

        def generate_frames_from_reps(self, reps, state=None):
            yield np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(
        quicktalk_runtime.QuickTalkRealtimeRuntime,
        "_worker_class",
        staticmethod(lambda: FakeWorker),
    )
    template = tmp_path / "template.mp4"
    template.write_bytes(b"template")
    service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    runtime = quicktalk_runtime.QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )
    monkeypatch.setattr(
        runtime,
        "_template_video_for",
        lambda session: Path(session.template_video or ""),
    )
    session = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"template_mode": "video", "template_video": str(template)},
    )

    first = runtime.preload_reference(session)
    second = runtime.preload_reference(session)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert build_count == 1

def test_wav2lip_init_accepts_frame_metadata_path(tmp_path: Path) -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    client.app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "ref_frame_metadata_path": str(metadata_path),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert "ref_frame_metadata_path" not in init


def test_wav2lip_frame_reference_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    client.app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[allowed])

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "reference_mode": "frames",
                "ref_frame_dir": str(outside),
            }
        )
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_frame_dir"
    assert str(outside) not in error["message"]


def test_wav2lip_preload_endpoint_uses_runtime_cache(tmp_path: Path) -> None:
    class FakePreloadRuntime:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def preload_reference(self, session):
            self.calls.append(session)
            return {
                "type": "preload_result",
                "frames": 2,
                "elapsed_ms": 12.5,
                "cache_hit": len(self.calls) > 1,
            }

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text('{"frames": {}}', encoding="utf-8")
    runtime = FakePreloadRuntime()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=runtime, allowed_frame_roots=[tmp_path])
    client = TestClient(app)
    payload = {
        "ref_frame_dir": str(frame_dir),
        "ref_frame_metadata_path": str(metadata_path),
        "width": 24,
        "height": 24,
        "fps": 30,
        "preprocessed": True,
        "wav2lip_postprocess_mode": "opentalking_improved",
    }

    first = client.post("/v1/audio2video/wav2lip/preload", json=payload)
    second = client.post("/v1/audio2video/wav2lip/preload", json=payload)

    assert first.status_code == 200
    assert first.json()["cache_hit"] is False
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert len(runtime.calls) == 2
    assert runtime.calls[0].reference_mode == "frames"
    assert runtime.calls[0].preprocessed is True


def test_wav2lip_preload_endpoint_offloads_runtime_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    class FakePreloadRuntime:
        def preload_reference(self, session):
            return {"type": "preload_result", "frames": 1, "elapsed_ms": 1.0, "cache_hit": False}

    calls = 0
    real_to_thread = avatar_routes.asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        nonlocal calls
        calls += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(avatar_routes.asyncio, "to_thread", tracking_to_thread)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=FakePreloadRuntime(),
        allowed_frame_roots=[tmp_path],
    )
    client = TestClient(app)

    response = client.post(
        "/v1/audio2video/wav2lip/preload",
        json={"ref_frame_dir": str(frame_dir), "width": 24, "height": 24},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "preload_result"
    assert calls == 1


def test_wav2lip_preload_endpoint_reports_runtime_error(tmp_path: Path) -> None:
    class FailingPreloadRuntime:
        def preload_reference(self, session):
            del session
            raise RuntimeError("preload failed")

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=FailingPreloadRuntime(),
        allowed_frame_roots=[tmp_path],
    )
    client = TestClient(app)

    response = client.post(
        "/v1/audio2video/wav2lip/preload",
        json={"ref_frame_dir": str(frame_dir), "width": 24, "height": 24},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "error"
    assert response.json()["code"] == "runtime_error"
    assert "preload failed" in response.json()["message"]
