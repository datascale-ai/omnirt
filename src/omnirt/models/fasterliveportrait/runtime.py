"""Realtime FasterLivePortrait/JoyVASA runtime owned by OmniRT.

The class is intentionally split into a protocol-safe lightweight path and a
real model-loading path. Unit tests and CPU protocol demos use the lightweight
path; production can enable model loading with ``OMNIRT_FASTLIVEPORTRAIT_RUNTIME``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
import ctypes
import io
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from omnirt.server.realtime_avatar import RealtimeAvatarError, RealtimeAvatarSession, encode_jpeg_sequence


log = logging.getLogger(__name__)
FASTLIVEPORTRAIT_MODEL_ID = "fasterliveportrait"
DEFAULT_CHECKPOINTS_DIR = Path("/data2/zhongyi/model/FasterLivePortrait/checkpoints")


@dataclass
class FasterLivePortraitSessionState:
    """Per-session streaming state for low-latency audio-to-motion rendering."""

    audio_buffer: bytearray = field(default_factory=bytearray)
    motion_history_frames: int = 0
    prev_motion_feat: Any | None = None
    prev_audio_feat: Any | None = None
    noise: Any | None = None
    emitted_frames: int = 0
    motion_cursor: int = 0
    source_path: str | None = None
    source_prepared: bool = False
    reference_frame: np.ndarray | None = None
    driving_frame_index: int = 0
    first_chunk_at: float = field(default_factory=time.monotonic)


class FasterLivePortraitRuntimeError(RuntimeError):
    """Raised when FasterLivePortrait cannot initialize or render."""


class FasterLivePortraitRealtimeRuntime:
    """Streaming FasterLivePortrait/JoyVASA runtime.

    The v1 runtime keeps model ownership in OmniRT and speaks the existing
    FlashTalk-compatible AUDI/VIDX protocol. ``load_models=False`` preserves the
    same wire contract with deterministic generated frames, which keeps tests
    fast and lets OpenTalking integrate before GPU model warmup is available.
    """

    def __init__(
        self,
        *,
        checkpoints_dir: str | Path | None = None,
        fasterliveportrait_root: str | Path | None = None,
        device: str | None = None,
        load_models: bool | None = None,
    ) -> None:
        self.checkpoints_dir = Path(
            checkpoints_dir
            or os.environ.get("OMNIRT_FASTLIVEPORTRAIT_CHECKPOINTS_DIR", "")
            or DEFAULT_CHECKPOINTS_DIR
        ).expanduser().resolve()
        self.fasterliveportrait_root = Path(
            fasterliveportrait_root
            or os.environ.get("OMNIRT_FASTLIVEPORTRAIT_ROOT", "")
            or "/data2/zhongyi/FasterLivePortrait"
        ).expanduser().resolve()
        self.device = device or os.environ.get("OMNIRT_FASTLIVEPORTRAIT_DEVICE", "cuda")
        self.load_models = self._parse_bool(
            os.environ.get("OMNIRT_FASTLIVEPORTRAIT_LOAD_MODELS"),
            default=False if load_models is None else bool(load_models),
        )
        self.work_root = Path(
            os.environ.get("OMNIRT_FASTLIVEPORTRAIT_WORK_DIR", "") or tempfile.gettempdir()
        ).expanduser().resolve()
        self.jpeg_quality = max(
            1,
            min(100, self._parse_int(os.environ.get("OMNIRT_FASTLIVEPORTRAIT_JPEG_QUALITY"), 85)),
        )
        self.default_emit_frames = max(
            1,
            self._parse_int(os.environ.get("OMNIRT_FASTLIVEPORTRAIT_EMIT_FRAMES_PER_CHUNK"), 12),
        )
        self.default_lookahead_ms = max(
            0,
            self._parse_int(os.environ.get("OMNIRT_FASTLIVEPORTRAIT_LOOKAHEAD_MS"), 320),
        )
        self._sessions: dict[str, FasterLivePortraitSessionState] = {}
        self._model_bundle: dict[str, Any] | None = None
        if self.load_models:
            self._validate_model_layout()

    def render_chunk(self, session: RealtimeAvatarSession, pcm_s16le: bytes) -> bytes:
        state = self._session_state(session)
        state.audio_buffer.extend(pcm_s16le)
        if self.load_models:
            return self._render_model_chunk(session, pcm_s16le, state)
        return self._render_placeholder_chunk(session, pcm_s16le, state)

    def render_driving_frame(self, session: RealtimeAvatarSession, frame_bytes: bytes) -> bytes:
        """Render one video-clone frame driven by a browser camera JPEG/PNG."""
        state = self._session_state(session)
        if self.load_models:
            return self._render_model_driving_frame(session, frame_bytes, state)
        return self._render_placeholder_driving_frame(session, frame_bytes, state)

    def preload_reference(self, session: RealtimeAvatarSession) -> dict[str, object]:
        """Load the model bundle and prepare the reference before live audio arrives."""
        started = time.monotonic()
        state = self._session_state(session)
        if self.load_models:
            bundle = self._load_model_bundle()
            with self._pushd(self.fasterliveportrait_root):
                pipeline = bundle["pipeline"]
                self._apply_runtime_config(pipeline, session)
                self._ensure_source_prepared(pipeline, session, state)
                self._reference_frame(session, state)
            warmed = True
        else:
            self._reference_frame(session, state)
            warmed = False
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        log.info(
            "FasterLivePortrait reference preloaded: session=%s warmed=%s elapsed_ms=%.1f",
            session.session_id,
            warmed,
            elapsed_ms,
        )
        return {
            "type": "preload_ok",
            "model": FASTLIVEPORTRAIT_MODEL_ID,
            "session_id": session.session_id,
            "elapsed_ms": elapsed_ms,
            "warmed": warmed,
        }

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def session_state(self, session_id: str) -> FasterLivePortraitSessionState:
        return self._sessions[session_id]

    def _session_state(self, session: RealtimeAvatarSession) -> FasterLivePortraitSessionState:
        existing = self._sessions.get(session.session_id)
        if existing is not None:
            return existing
        state = FasterLivePortraitSessionState()
        self._sessions[session.session_id] = state
        return state

    def _gpu_memory_mb(self) -> int:
        try:
            import torch

            if not torch.cuda.is_available():
                return -1
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return -1

    def _render_model_chunk(
        self,
        session: RealtimeAvatarSession,
        pcm_s16le: bytes,
        state: FasterLivePortraitSessionState,
    ) -> bytes:
        """Render one low-latency audio chunk with JoyVASA + FasterLivePortrait."""

        chunk_t0 = time.monotonic()
        bundle = self._load_model_bundle()
        motion_t0 = time.monotonic()
        motion_infos = self._generate_motion_infos(bundle["joyvasa"], pcm_s16le, state, session)
        motion_ms = (time.monotonic() - motion_t0) * 1000.0
        if not motion_infos:
            return self._render_placeholder_chunk(session, pcm_s16le, state)

        emit_frames = self._emit_frames(session)
        if self._bool_config(session, "disable_frame_interpolation", False):
            selected_motion_infos = self._limit_raw_motion_infos(motion_infos, emit_frames)
            selected_lip_ratios = self._lip_ratios_for_chunk(
                pcm_s16le,
                session.audio.sample_rate,
                len(selected_motion_infos),
                session,
            )
        else:
            motion_infos = self._resample_motion_infos(motion_infos, emit_frames)
            lip_ratios = self._lip_ratios_for_chunk(pcm_s16le, session.audio.sample_rate, emit_frames, session)
            render_keyframes = self._render_keyframes(session, emit_frames)
            selected_motion_infos = self._select_keyframe_motions(motion_infos, render_keyframes)
            selected_lip_ratios = self._select_keyframe_scalars(lip_ratios, render_keyframes)
        pipeline = bundle["pipeline"]
        keyframe_jpegs: list[bytes] = []
        prepare_ms = 0.0
        render_ms = 0.0
        compose_ms = 0.0
        encode_ms = 0.0
        render_t0 = time.monotonic()
        with self._pushd(self.fasterliveportrait_root):
            prepare_t0 = time.monotonic()
            self._apply_runtime_config(pipeline, session)
            self._ensure_source_prepared(pipeline, session, state)
            prepare_ms = (time.monotonic() - prepare_t0) * 1000.0
            for offset, (motion_info, lip_ratio) in enumerate(zip(selected_motion_infos, selected_lip_ratios)):
                frame_t0 = time.monotonic()
                out_crop, out_org = self._run_realtime_motion(
                    pipeline,
                    motion_info,
                    first_frame=state.emitted_frames == 0 and offset == 0,
                    session=session,
                    lip_ratio=lip_ratio,
                )
                render_ms += (time.monotonic() - frame_t0) * 1000.0
                frame = self._select_output_frame(out_crop, out_org, session)
                if frame is None:
                    continue
                compose_t0 = time.monotonic()
                frame = self._compose_head_on_reference(frame, session, state)
                compose_ms += (time.monotonic() - compose_t0) * 1000.0
                encode_t0 = time.monotonic()
                keyframe_jpegs.append(self._encode_rgb_jpeg(frame, session))
                encode_ms += (time.monotonic() - encode_t0) * 1000.0
        if self._bool_config(session, "disable_frame_interpolation", False):
            jpeg_frames = keyframe_jpegs
        else:
            expand_t0 = time.monotonic()
            jpeg_frames = self._expand_keyframes(keyframe_jpegs, emit_frames)
            encode_ms += (time.monotonic() - expand_t0) * 1000.0
        render_encode_ms = (time.monotonic() - render_t0) * 1000.0

        state.emitted_frames += len(jpeg_frames)
        state.motion_cursor += len(jpeg_frames)
        if not jpeg_frames:
            return self._render_placeholder_chunk(session, pcm_s16le, state)
        payload = encode_jpeg_sequence(jpeg_frames)
        total_ms = (time.monotonic() - chunk_t0) * 1000.0
        payload_kb = len(payload) / 1024.0
        message = (
            "FasterLivePortrait chunk rendered: "
            f"session={session.session_id} frames={len(jpeg_frames)} "
            f"selected={len(selected_motion_infos)} motion_ms={motion_ms:.1f} "
            f"prepare_ms={prepare_ms:.1f} render_ms={render_ms:.1f} "
            f"compose_ms={compose_ms:.1f} encode_ms={encode_ms:.1f} "
            f"render_encode_ms={render_encode_ms:.1f} total_ms={total_ms:.1f} "
            f"fps={len(jpeg_frames) / max(total_ms / 1000.0, 1e-6):.2f} "
            f"payload_kb={payload_kb:.0f} "
            f"kb_per_frame={payload_kb / max(1, len(jpeg_frames)):.1f} "
            f"gpu_mem_mb={self._gpu_memory_mb()}"
        )
        log.info(message)
        print(message, flush=True)
        return payload

    def _render_placeholder_chunk(
        self,
        session: RealtimeAvatarSession,
        pcm_s16le: bytes,
        state: FasterLivePortraitSessionState,
    ) -> bytes:
        frame_count = self._emit_frames(session)
        width = max(16, int(session.video.width))
        height = max(16, int(session.video.height))
        energy = self._audio_energy(pcm_s16le)
        head = self._float_config(session, "head_motion_multiplier", 0.45)
        expr = self._float_config(session, "expression_multiplier", 1.0)
        frames: list[bytes] = []
        for idx in range(frame_count):
            frame_no = state.emitted_frames + idx
            frames.append(
                self._placeholder_jpeg(
                    width=width,
                    height=height,
                    frame_no=frame_no,
                    energy=energy,
                    head_multiplier=head,
                    expression_multiplier=expr,
                )
            )
        state.emitted_frames += frame_count
        state.motion_cursor += frame_count
        return encode_jpeg_sequence(frames)

    def _render_placeholder_driving_frame(
        self,
        session: RealtimeAvatarSession,
        frame_bytes: bytes,
        state: FasterLivePortraitSessionState,
    ) -> bytes:
        driving = self._decode_rgb_image(frame_bytes)
        if driving is None:
            raise RealtimeAvatarError("bad_frame", "Driving frame must be a valid JPEG or PNG image.")
        frame_no = state.driving_frame_index
        state.driving_frame_index += 1
        state.emitted_frames += 1
        energy = float(np.std(driving.astype(np.float32)) / 128.0)
        jpeg = self._placeholder_jpeg(
            width=max(16, int(session.video.width)),
            height=max(16, int(session.video.height)),
            frame_no=frame_no,
            energy=min(1.0, max(0.0, energy)),
            head_multiplier=self._float_config(session, "head_motion_multiplier", 0.45),
            expression_multiplier=self._float_config(session, "expression_multiplier", 1.0),
        )
        return encode_jpeg_sequence([jpeg])

    def _render_model_driving_frame(
        self,
        session: RealtimeAvatarSession,
        frame_bytes: bytes,
        state: FasterLivePortraitSessionState,
    ) -> bytes:
        driving_rgb = self._decode_rgb_image(frame_bytes)
        if driving_rgb is None:
            raise RealtimeAvatarError("bad_frame", "Driving frame must be a valid JPEG or PNG image.")
        bundle = self._load_model_bundle()
        pipeline = bundle["pipeline"]
        with self._pushd(self.fasterliveportrait_root):
            self._apply_runtime_config(pipeline, session)
            self._ensure_source_prepared(pipeline, session, state)
            driving_bgr = driving_rgb[:, :, ::-1].copy()
            try:
                dri_crop, out_crop, out_org, _dri_motion_info = pipeline.run(
                    driving_bgr,
                    pipeline.src_imgs[0],
                    pipeline.src_infos[0],
                    first_frame=state.driving_frame_index == 0,
                )
            except Exception as exc:
                raise FasterLivePortraitRuntimeError(f"FasterLivePortrait video-clone render failed: {exc}") from exc
        if dri_crop is None and out_crop is None and out_org is None:
            raise RealtimeAvatarError("no_driving_face", "No face was detected in the driving frame.")
        frame = self._select_output_frame(out_crop, out_org, session)
        if frame is None:
            raise RealtimeAvatarError("no_driving_face", "No output frame was produced from the driving frame.")
        state.driving_frame_index += 1
        state.emitted_frames += 1
        return encode_jpeg_sequence([self._encode_rgb_jpeg(frame, session)])

    def _run_realtime_motion(
        self,
        pipeline: Any,
        motion_info: dict[str, Any],
        *,
        first_frame: bool,
        session: RealtimeAvatarSession,
        lip_ratio: float | None = None,
    ) -> tuple[Any, Any]:
        import copy
        import torch

        x_d_i_info = self._apply_pose_motion_multiplier(motion_info, session)
        R_d_i = x_d_i_info["R"] if "R" in x_d_i_info else x_d_i_info["R_d"]
        if first_frame or getattr(pipeline, "R_d_0", None) is None:
            pipeline.frame_id = 0
            pipeline.R_d_0 = copy.deepcopy(R_d_i)
            pipeline.x_d_0_info = copy.deepcopy(x_d_i_info)
            try:
                from src.utils import utils as flp_utils

                pipeline.R_d_smooth = flp_utils.OneEuroFilter(4, 0.3)
                pipeline.exp_smooth = flp_utils.OneEuroFilter(4, 0.3)
            except ModuleNotFoundError:
                pass
        device = getattr(pipeline, "device", "cpu")
        I_p_pstbk = torch.from_numpy(pipeline.src_imgs[0]).to(device).float()
        realtime = False if self._bool_config(session, "flag_stitching", False) else True
        return pipeline._run(
            pipeline.src_infos[0],
            x_d_i_info,
            copy.deepcopy(pipeline.x_d_0_info),
            R_d_i,
            copy.deepcopy(pipeline.R_d_0),
            realtime,
            None,
            lip_ratio,
            I_p_pstbk,
        )

    def _apply_pose_motion_multiplier(
        self,
        motion_info: dict[str, Any],
        session: RealtimeAvatarSession,
    ) -> dict[str, Any]:
        multiplier = self._float_config(session, "pose_motion_multiplier", 1.0)
        if abs(multiplier - 1.0) < 1e-6:
            return motion_info
        import copy
        from src.utils import utils as flp_utils

        out = copy.deepcopy(motion_info)
        for key in ("pitch", "yaw", "roll"):
            if key not in out:
                return motion_info
        pitch = np.asarray(out["pitch"], dtype=np.float32) * multiplier
        yaw = np.asarray(out["yaw"], dtype=np.float32) * multiplier
        roll = np.asarray(out["roll"], dtype=np.float32) * multiplier
        out["pitch"] = pitch.astype(np.float32, copy=False)
        out["yaw"] = yaw.astype(np.float32, copy=False)
        out["roll"] = roll.astype(np.float32, copy=False)
        out["R"] = flp_utils.get_rotation_matrix(pitch, yaw, roll).reshape(1, 3, 3).astype(np.float32)
        if "R_d" in out:
            out["R_d"] = out["R"]
        return out

    def _select_output_frame(self, out_crop: Any, out_org: Any, session: RealtimeAvatarSession) -> Any:
        """Choose the frame that actually contains motion.

        FasterLivePortrait returns ``I_p_pstbk`` as the second value. In realtime
        mode that tensor is just the unchanged source canvas, so sending it to
        OpenTalking produces a valid but visually static video. When pasteback is
        disabled or unavailable, prefer the animated crop so the browser receives
        visible motion instead of repeated source frames.
        """
        if (
            out_org is not None
            and self._bool_config(session, "flag_stitching", False)
            and self._bool_config(session, "flag_pasteback", True)
        ):
            return out_org
        return out_crop if out_crop is not None else out_org

    def _compose_head_on_reference(
        self,
        frame: Any,
        session: RealtimeAvatarSession,
        state: FasterLivePortraitSessionState,
    ) -> np.ndarray:
        if not self._bool_config(session, "head_only_pasteback", True):
            return np.asarray(frame).astype(np.uint8, copy=False)
        reference = self._reference_frame(session, state)
        if reference is None:
            return np.asarray(frame).astype(np.uint8, copy=False)
        generated = np.asarray(frame).astype(np.uint8, copy=False)
        if generated.shape != reference.shape:
            generated = np.asarray(
                Image.fromarray(generated).resize(
                    (reference.shape[1], reference.shape[0]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
        mask = self._head_only_mask(reference, generated)
        if mask is None:
            return generated
        alpha = mask[:, :, None].astype(np.float32)
        out = generated.astype(np.float32) * alpha + reference.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _reference_frame(
        self,
        session: RealtimeAvatarSession,
        state: FasterLivePortraitSessionState,
    ) -> np.ndarray | None:
        if state.reference_frame is not None:
            return state.reference_frame
        try:
            image = Image.open(io.BytesIO(session.image_bytes)).convert("RGB")
        except Exception:
            return None
        target = (int(session.video.width), int(session.video.height))
        if image.size != target:
            image = image.resize(target, Image.Resampling.BILINEAR)
        state.reference_frame = np.asarray(image, dtype=np.uint8)
        return state.reference_frame

    def _head_only_mask(self, reference: np.ndarray, generated: np.ndarray) -> np.ndarray | None:
        diff = np.max(
            np.abs(generated.astype(np.int16) - reference.astype(np.int16)),
            axis=2,
        )
        height, width = diff.shape
        head_limit = max(1, int(height * 0.68))
        diff[head_limit:, :] = 0
        ys, xs = np.where(diff > 6)
        if xs.size == 0 or ys.size == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        box_w = max(1, x2 - x1 + 1)
        box_h = max(1, y2 - y1 + 1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        rx = max(box_w * 0.82, width * 0.13)
        ry = max(box_h * 0.78, height * 0.16)
        top = max(0.0, cy - ry * 1.02)
        bottom = min(float(head_limit), cy + ry * 0.92)
        left = max(0.0, cx - rx)
        right = min(float(width - 1), cx + rx)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        ellipse = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2
        mask = (ellipse <= 1.0).astype(np.float32)
        # Keep the lower face fully animated so pasteback does not wash out mouth openings.
        mouth_top = max(top, cy + ry * 0.05)
        mouth_bottom = min(bottom, cy + ry * 0.58)
        mouth_left = max(left, cx - rx * 0.48)
        mouth_right = min(right, cx + rx * 0.48)
        mouth_region = (
            (yy >= mouth_top)
            & (yy <= mouth_bottom)
            & (xx >= mouth_left)
            & (xx <= mouth_right)
        )
        mask[mouth_region] = 1.0
        mask[(yy < top) | (yy > bottom) | (xx < left) | (xx > right)] = 0.0
        if mask.max() <= 0:
            return None
        for _ in range(5):
            padded = np.pad(mask, 1, mode="edge")
            mask = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + padded[1:-1, 1:-1]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 9.0
            mask[(yy < top) | (yy > bottom) | (xx < left) | (xx > right)] = 0.0
        return np.clip(mask, 0.0, 1.0)

    def _load_model_bundle(self) -> dict[str, Any]:
        if self._model_bundle is not None:
            return self._model_bundle
        self._validate_model_layout()
        import sys

        root = str(self.fasterliveportrait_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        with self._pushd(self.fasterliveportrait_root):
            from omegaconf import OmegaConf
            from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
            from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

            cfg_path = Path(os.environ.get("OMNIRT_FASTLIVEPORTRAIT_CFG", "configs/onnx_infer.yaml"))
            if not cfg_path.is_absolute():
                cfg_path = self.fasterliveportrait_root / cfg_path
            self._validate_runtime_dependencies(cfg_path)
            cfg = OmegaConf.load(cfg_path)
            self._absolutize_flp_config(cfg)
            pipeline = GradioLivePortraitPipeline(cfg)
            self._force_eager_attention_for_joyvasa_audio_encoders()
            joyvasa = JoyVASAAudio2MotionPipeline(
                motion_model_path=str(self.checkpoints_dir / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese.pt"),
                audio_model_path=str(self.checkpoints_dir / "chinese-hubert-base"),
                motion_template_path=str(self.checkpoints_dir / "JoyVASA" / "motion_template" / "motion_template.pkl"),
                cfg_mode=str(getattr(cfg.infer_params, "cfg_mode", "incremental")),
                cfg_scale=float(getattr(cfg.infer_params, "cfg_scale", 3.5)),
            )
        self._model_bundle = {
            "checkpoints_dir": self.checkpoints_dir,
            "fasterliveportrait_root": self.fasterliveportrait_root,
            "pipeline": pipeline,
            "joyvasa": joyvasa,
        }
        return self._model_bundle

    def _force_eager_attention_for_joyvasa_audio_encoders(self) -> None:
        from src.models.JoyVASA.hubert import HubertModel
        from src.models.JoyVASA.wav2vec2 import Wav2Vec2Model

        self._force_eager_attention_for_audio_encoder(HubertModel, "HuBERT")
        self._force_eager_attention_for_audio_encoder(Wav2Vec2Model, "Wav2Vec2")

    def _force_eager_attention_for_audio_encoder(self, model_cls: Any, model_name: str) -> None:
        current = getattr(model_cls, "from_pretrained")
        current_func = getattr(current, "__func__", current)
        if getattr(current_func, "_omnirt_forces_eager_attention", False):
            return

        original_from_pretrained = current

        def from_pretrained_with_eager(cls: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs["attn_implementation"] = "eager"
            model = original_from_pretrained(*args, **kwargs)
            config = getattr(model, "config", None)
            if config is not None:
                setattr(config, "_attn_implementation", "eager")
            log.info("JoyVASA %s audio encoder loaded with eager attention", model_name)
            return model

        from_pretrained_with_eager._omnirt_forces_eager_attention = True  # type: ignore[attr-defined]
        model_cls.from_pretrained = classmethod(from_pretrained_with_eager)

    def _validate_model_layout(self) -> None:
        required = (
            self.checkpoints_dir / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese.pt",
            self.checkpoints_dir / "JoyVASA" / "motion_template" / "motion_template.pkl",
            self.checkpoints_dir / "chinese-hubert-base" / "config.json",
            self.checkpoints_dir / "liveportrait_onnx",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FasterLivePortraitRuntimeError(
                "FasterLivePortrait checkpoints are incomplete: " + ", ".join(missing)
            )

    def _validate_runtime_dependencies(self, cfg_path: Path) -> None:
        cfg_name = str(cfg_path).lower()
        if "trt" not in cfg_name:
            return
        if importlib.util.find_spec("tensorrt") is None:
            raise FasterLivePortraitRuntimeError(
                "TensorRT is required for FasterLivePortrait TRT config, but the current OmniRT venv "
                "cannot import tensorrt. Install the TensorRT Python wheels in the worktree venv "
                "or set OMNIRT_FASTLIVEPORTRAIT_CFG=configs/onnx_infer.yaml."
            )
        self._load_tensorrt_plugins()

    def _load_tensorrt_plugins(self) -> None:
        plugin_path = self.checkpoints_dir / "liveportrait_onnx" / "libgrid_sample_3d_plugin.so"
        if not plugin_path.exists():
            raise FasterLivePortraitRuntimeError(
                "GridSample3D TensorRT plugin is required for FasterLivePortrait TRT config, "
                f"but was not found at {plugin_path}."
            )
        try:
            ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise FasterLivePortraitRuntimeError(
                "GridSample3D TensorRT plugin is required for FasterLivePortrait TRT config, "
                f"but failed to load {plugin_path}: {exc}"
            ) from exc
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        registry = trt.get_plugin_registry()
        creators = getattr(registry, "plugin_creator_list", ())
        if not any(getattr(creator, "name", "") == "GridSample3D" for creator in creators):
            raise FasterLivePortraitRuntimeError(
                "GridSample3D TensorRT plugin is required for FasterLivePortrait TRT config, "
                f"but loading {plugin_path} did not register the plugin."
            )

    @contextmanager
    def _pushd(self, path: Path):
        old_cwd = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _absolutize_flp_config(self, cfg: Any) -> None:
        def rewrite(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(("./checkpoints/", "checkpoints/")):
                suffix = value.replace("./checkpoints/", "", 1).replace("checkpoints/", "", 1)
                return str(self.checkpoints_dir / suffix)
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            return value

        for section_name in ("models", "animal_models"):
            section = getattr(cfg, section_name, None)
            if not section:
                continue
            for item in section.values():
                if "model_path" in item:
                    item["model_path"] = rewrite(item["model_path"])
        cfg.joyvasa_models.motion_model_path = str(
            self.checkpoints_dir / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese.pt"
        )
        cfg.joyvasa_models.audio_model_path = str(self.checkpoints_dir / "chinese-hubert-base")
        cfg.joyvasa_models.motion_template_path = str(
            self.checkpoints_dir / "JoyVASA" / "motion_template" / "motion_template.pkl"
        )

    def _ensure_source_prepared(
        self,
        pipeline: Any,
        session: RealtimeAvatarSession,
        state: FasterLivePortraitSessionState,
    ) -> None:
        if state.source_prepared and state.source_path == pipeline.source_path:
            return
        self.work_root.mkdir(parents=True, exist_ok=True)
        source_path = self.work_root / f"omnirt_flp_{session.session_id}.png"
        if state.source_path is None:
            source_path.write_bytes(session.image_bytes)
            state.source_path = str(source_path)
        pipeline.init_vars()
        if not pipeline.prepare_source(state.source_path, realtime=True):
            raise FasterLivePortraitRuntimeError("FasterLivePortrait failed to prepare the reference image.")
        state.source_prepared = True

    def _apply_runtime_config(self, pipeline: Any, session: RealtimeAvatarSession) -> None:
        args: dict[str, Any] = {}
        for key in (
            "head_motion_multiplier",
            "expression_multiplier",
            "yaw_multiplier",
            "pitch_multiplier",
            "roll_multiplier",
            "animation_region",
            "mouth_open_multiplier",
            "mouth_corner_multiplier",
            "cheek_jaw_multiplier",
            "driving_multiplier",
            "cfg_scale",
            "flag_stitching",
            "flag_pasteback",
            "flag_relative_motion",
            "flag_crop_driving_video",
            "flag_normalize_lip",
            "flag_lip_retargeting",
            "lip_retargeting_multiplier",
            "lip_retargeting_min",
            "lip_retargeting_max",
            "lip_retargeting_noise_floor",
            "render_keyframes_per_chunk",
            "disable_frame_interpolation",
        ):
            if key in session.runtime_config:
                args[key] = session.runtime_config[key]
        if args:
            pipeline.update_cfg(args)

    def _lip_ratios_for_chunk(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        frame_count: int,
        session: RealtimeAvatarSession,
    ) -> list[float | None]:
        if not self._bool_config(session, "flag_lip_retargeting", False):
            return [None] * max(1, frame_count)
        pcm = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size == 0:
            return [self._float_config(session, "lip_retargeting_min", 0.02)] * max(1, frame_count)
        floor = max(0.0, self._float_config(session, "lip_retargeting_noise_floor", 0.012))
        gain = max(0.0, self._float_config(session, "lip_retargeting_multiplier", 3.2))
        min_ratio = max(0.0, self._float_config(session, "lip_retargeting_min", 0.02))
        max_ratio = max(min_ratio, self._float_config(session, "lip_retargeting_max", 0.62))
        out: list[float | None] = []
        for idx in range(max(1, frame_count)):
            start = round(idx * pcm.size / float(max(1, frame_count)))
            end = round((idx + 1) * pcm.size / float(max(1, frame_count)))
            window = pcm[start:max(end, start + 1)]
            rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
            voiced = max(0.0, rms - floor)
            ratio = min_ratio + voiced * gain
            out.append(float(np.clip(ratio, min_ratio, max_ratio)))
        return out

    def _generate_motion_infos(
        self,
        joyvasa: Any,
        pcm_s16le: bytes,
        state: FasterLivePortraitSessionState,
        session: RealtimeAvatarSession,
    ) -> list[dict[str, Any]]:
        import numpy as np
        import torch
        import torch.nn.functional as F

        pcm = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size == 0:
            return []
        target_samples = int(joyvasa.n_audio_samples)
        chunk_samples = int(pcm.size)
        clip_len = max(1, int(chunk_samples / float(session.audio.sample_rate) * joyvasa.fps))
        clip_len = min(clip_len, int(joyvasa.n_motions))
        max_buffer_bytes = max(target_samples, chunk_samples) * 2
        if len(state.audio_buffer) > max_buffer_bytes:
            del state.audio_buffer[:-max_buffer_bytes]
        audio_bytes = bytes(state.audio_buffer) or pcm_s16le
        buffered = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if buffered.size == 0:
            return []
        if buffered.size >= target_samples:
            window = buffered[-target_samples:]
            valid_frames = int(joyvasa.n_motions)
        else:
            n_padding_audio_samples = target_samples - int(buffered.size)
            padding_value = 0.0 if joyvasa.pad_mode == "zero" else float(buffered[0])
            window = np.pad(buffered, (n_padding_audio_samples, 0), constant_values=padding_value)
            valid_frames = max(1, int(math.ceil(buffered.size / joyvasa.audio_unit)))
        audio = torch.from_numpy(window.astype(np.float32, copy=False)).to(joyvasa.device, dtype=joyvasa.dtype)
        audio_in = audio[:target_samples].unsqueeze(0)
        indicator = torch.ones((1, joyvasa.n_motions), device=joyvasa.device) if joyvasa.use_indicator else None
        if indicator is not None:
            invalid_frames = max(0, int(joyvasa.n_motions) - valid_frames)
            if invalid_frames > 0:
                indicator[:, :invalid_frames] = 0
        cfg_scale = self._float_config(session, "cfg_scale", float(joyvasa.cfg_scale))
        cfg_cond = joyvasa.cfg_cond
        if cfg_scale <= 0:
            cfg_cond = []
        with torch.inference_mode():
            if state.prev_motion_feat is None or state.prev_audio_feat is None or state.noise is None:
                motion_feat, noise, prev_audio_feat = joyvasa.motion_generator.sample(
                    audio_in,
                    indicator=indicator,
                    cfg_mode=joyvasa.cfg_mode,
                    cfg_cond=cfg_cond,
                    cfg_scale=cfg_scale,
                    dynamic_threshold=0,
                )
            else:
                motion_feat, noise, prev_audio_feat = joyvasa.motion_generator.sample(
                    audio_in,
                    state.prev_motion_feat.to(joyvasa.dtype),
                    state.prev_audio_feat.to(joyvasa.dtype),
                    state.noise.to(joyvasa.dtype),
                    indicator=indicator,
                    cfg_mode=joyvasa.cfg_mode,
                    cfg_cond=cfg_cond,
                    cfg_scale=cfg_scale,
                    dynamic_threshold=0,
                )
        state.prev_motion_feat = motion_feat[:, -joyvasa.n_prev_motions:].clone()
        state.prev_audio_feat = prev_audio_feat[:, -joyvasa.n_prev_motions:].clone()
        state.noise = noise.clone() if hasattr(noise, "clone") else noise
        start = max(0, int(motion_feat.shape[1]) - clip_len)
        motion_coef = motion_feat[:, start:start + clip_len].squeeze(0).detach().cpu().numpy().astype(np.float32)
        state.motion_history_frames += int(motion_coef.shape[0])
        return [self._motion_coef_to_motion(joyvasa, coef) for coef in motion_coef]

    def _motion_coef_to_motion(self, joyvasa: Any, coef: Any) -> dict[str, Any]:
        import numpy as np
        from src.utils import utils as flp_utils

        template = joyvasa.templete_dict
        exp = coef[:63] * template["std_exp"] + template["mean_exp"]
        scale = coef[63:64] * (template["max_scale"] - template["min_scale"]) + template["min_scale"]
        t = coef[64:67] * (template["max_t"] - template["min_t"]) + template["min_t"]
        pitch = coef[67:68] * (template["max_pitch"] - template["min_pitch"]) + template["min_pitch"]
        yaw = coef[68:69] * (template["max_yaw"] - template["min_yaw"]) + template["min_yaw"]
        roll = coef[69:70] * (template["max_roll"] - template["min_roll"]) + template["min_roll"]
        r = flp_utils.get_rotation_matrix(pitch, yaw, roll).reshape(1, 3, 3).astype(np.float32)
        return {
            "exp": exp.reshape(1, 21, 3).astype(np.float32),
            "scale": scale.reshape(1, 1).astype(np.float32),
            "R": r,
            "t": t.reshape(1, 3).astype(np.float32),
            "pitch": pitch.reshape(1, 1).astype(np.float32),
            "yaw": yaw.reshape(1, 1).astype(np.float32),
            "roll": roll.reshape(1, 1).astype(np.float32),
        }

    def _encode_rgb_jpeg(self, frame: Any, session: RealtimeAvatarSession) -> bytes:
        import numpy as np

        arr = np.asarray(frame).astype(np.uint8, copy=False)
        image = Image.fromarray(arr)
        target = (int(session.video.width), int(session.video.height))
        if image.size != target:
            image = image.resize(target, Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        return buffer.getvalue()

    def _decode_rgb_image(self, frame_bytes: bytes) -> np.ndarray | None:
        try:
            image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        except Exception:
            return None
        return np.asarray(image, dtype=np.uint8)

    def _emit_frames(self, session: RealtimeAvatarSession) -> int:
        raw = session.runtime_config.get("emit_frames_per_chunk")
        if raw is None:
            raw = session.video.slice_len or self.default_emit_frames
        return max(1, int(raw))

    def _render_keyframes(self, session: RealtimeAvatarSession, emit_frames: int) -> int:
        raw = session.runtime_config.get("render_keyframes_per_chunk")
        if raw is None:
            raw = os.environ.get("OMNIRT_FASTLIVEPORTRAIT_RENDER_KEYFRAMES_PER_CHUNK", "6")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 6
        return max(1, min(emit_frames, value))

    def _limit_raw_motion_infos(self, motion_infos: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
        if max_frames <= 0 or not motion_infos:
            return []
        if len(motion_infos) <= max_frames:
            return list(motion_infos)
        return self._select_keyframe_motions(motion_infos, max_frames)

    def _select_keyframe_motions(self, motion_infos: list[dict[str, Any]], keyframes: int) -> list[dict[str, Any]]:
        if keyframes >= len(motion_infos):
            return list(motion_infos)
        if keyframes <= 1:
            return [motion_infos[0]]
        last = len(motion_infos) - 1
        indices = [round(i * last / float(keyframes - 1)) for i in range(keyframes)]
        return [motion_infos[idx] for idx in indices]

    def _select_keyframe_scalars(self, values: list[float | None], keyframes: int) -> list[float | None]:
        if not values:
            return [None] * max(1, keyframes)
        if keyframes >= len(values):
            return list(values)
        if keyframes <= 1:
            return [values[0]]
        last = len(values) - 1
        indices = [round(i * last / float(keyframes - 1)) for i in range(keyframes)]
        return [values[idx] for idx in indices]

    def _resample_motion_infos(self, motion_infos: list[dict[str, Any]], emit_frames: int) -> list[dict[str, Any]]:
        """Resample sparse JoyVASA motions to the browser frame cadence."""

        if emit_frames <= 0 or not motion_infos:
            return []
        if len(motion_infos) == emit_frames:
            return list(motion_infos)
        if len(motion_infos) == 1:
            return [self._copy_motion_info(motion_infos[0]) for _ in range(emit_frames)]

        last = len(motion_infos) - 1
        out: list[dict[str, Any]] = []
        for idx in range(emit_frames):
            pos = idx * last / float(max(emit_frames - 1, 1))
            lo = int(math.floor(pos))
            hi = min(last, lo + 1)
            weight = float(pos - lo)
            out.append(self._interpolate_motion_info(motion_infos[lo], motion_infos[hi], weight))
        return out

    def _interpolate_motion_info(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        weight: float,
    ) -> dict[str, Any]:
        if weight <= 0:
            return self._copy_motion_info(left)
        if weight >= 1:
            return self._copy_motion_info(right)

        import copy
        import numpy as np

        blended: dict[str, Any] = {}
        keys = set(left) | set(right)
        for key in keys:
            if key not in left:
                blended[key] = copy.deepcopy(right[key])
                continue
            if key not in right:
                blended[key] = copy.deepcopy(left[key])
                continue
            l_value = left[key]
            r_value = right[key]
            try:
                l_arr = np.asarray(l_value)
                r_arr = np.asarray(r_value)
                if l_arr.shape == r_arr.shape and np.issubdtype(l_arr.dtype, np.number) and np.issubdtype(r_arr.dtype, np.number):
                    mixed = l_arr * (1.0 - weight) + r_arr * weight
                    blended[key] = mixed.astype(l_arr.dtype, copy=False) if hasattr(l_value, "dtype") else mixed
                    continue
            except (TypeError, ValueError):
                pass
            blended[key] = copy.deepcopy(l_value if weight < 0.5 else r_value)
        return blended

    @staticmethod
    def _copy_motion_info(motion_info: dict[str, Any]) -> dict[str, Any]:
        import copy

        return copy.deepcopy(motion_info)

    def _expand_keyframes(self, keyframe_jpegs: list[bytes], emit_frames: int) -> list[bytes]:
        if not keyframe_jpegs:
            return []
        if len(keyframe_jpegs) >= emit_frames:
            return keyframe_jpegs[:emit_frames]
        keyframe_jpegs = self._dedupe_consecutive_jpegs(keyframe_jpegs)
        if len(keyframe_jpegs) >= emit_frames:
            return keyframe_jpegs[:emit_frames]
        if len(keyframe_jpegs) > 1:
            blended = self._interpolate_jpeg_frames(keyframe_jpegs, emit_frames)
            if blended:
                return blended
        last = emit_frames - 1
        key_last = len(keyframe_jpegs) - 1
        return [keyframe_jpegs[round(i * key_last / float(last))] for i in range(emit_frames)]

    def _dedupe_consecutive_jpegs(self, frames: list[bytes]) -> list[bytes]:
        out: list[bytes] = []
        previous: bytes | None = None
        for frame in frames:
            if frame != previous:
                out.append(frame)
            previous = frame
        return out

    def _interpolate_jpeg_frames(self, keyframe_jpegs: list[bytes], emit_frames: int) -> list[bytes]:
        decoded: list[np.ndarray] = []
        for frame in keyframe_jpegs:
            try:
                decoded.append(np.asarray(Image.open(io.BytesIO(frame)).convert("RGB"), dtype=np.float32))
            except Exception:
                return []
        if not decoded:
            return []
        if len(decoded) == 1:
            return [keyframe_jpegs[0] for _ in range(emit_frames)]

        out: list[bytes] = []
        key_last = len(decoded) - 1
        for idx in range(emit_frames):
            pos = idx * key_last / float(max(emit_frames - 1, 1))
            lo = int(math.floor(pos))
            hi = min(key_last, lo + 1)
            weight = float(pos - lo)
            if weight <= 0:
                arr = decoded[lo]
            elif hi == lo:
                arr = decoded[lo]
            else:
                arr = decoded[lo] * (1.0 - weight) + decoded[hi] * weight
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality)
            out.append(buffer.getvalue())
        return out

    def _placeholder_jpeg(
        self,
        *,
        width: int,
        height: int,
        frame_no: int,
        energy: float,
        head_multiplier: float,
        expression_multiplier: float,
    ) -> bytes:
        bg = (
            int(32 + min(160, energy * 220)),
            int((frame_no * 7) % 96 + 48),
            int(120 + min(100, expression_multiplier * 32)),
        )
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        cx = width // 2 + int(math.sin(frame_no / 4.0) * head_multiplier * width * 0.08)
        cy = height // 2
        radius = max(8, min(width, height) // 5)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(235, 225, 208))
        mouth_w = int(radius * 0.9)
        mouth_h = max(2, int(radius * (0.12 + energy * 0.28) * expression_multiplier))
        draw.ellipse((cx - mouth_w // 2, cy + radius // 4, cx + mouth_w // 2, cy + radius // 4 + mouth_h), fill=(78, 32, 42))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        return buffer.getvalue()

    @staticmethod
    def _audio_energy(pcm_s16le: bytes) -> float:
        if not pcm_s16le:
            return 0.0
        total = 0
        count = 0
        for idx in range(0, len(pcm_s16le) - 1, 2):
            sample = int.from_bytes(pcm_s16le[idx : idx + 2], "little", signed=True)
            total += abs(sample)
            count += 1
        if count == 0:
            return 0.0
        return min(1.0, total / float(count * 32768))

    @staticmethod
    def _float_config(session: RealtimeAvatarSession, key: str, default: float) -> float:
        raw = session.runtime_config.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool_config(session: RealtimeAvatarSession, key: str, default: bool) -> bool:
        raw = session.runtime_config.get(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    @staticmethod
    def _parse_int(raw: str | None, default: int) -> int:
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _parse_bool(raw: str | None, *, default: bool) -> bool:
        if raw is None or not raw.strip():
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}
