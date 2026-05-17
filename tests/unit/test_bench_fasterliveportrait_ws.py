from __future__ import annotations

from scripts import bench_fasterliveportrait_ws as bench


def test_parse_args_derives_chunks_and_emit_frames_from_duration() -> None:
    args = bench.parse_args(["--duration", "30", "--chunk-samples", "16000"])

    assert args.chunks == 30
    assert args.emit_frames == 25
    assert args.head_motion_multiplier == 0.3
    assert args.yaw_multiplier == 0.85
    assert args.pitch_multiplier == 1.0
    assert args.roll_multiplier == 0.85
    assert args.animation_region == "lip"
    assert args.mouth_open_multiplier == 1.25
    assert args.expression_multiplier == 1.0
    assert args.mouth_corner_multiplier == 0.85
    assert args.cheek_jaw_multiplier == 0.9
    assert args.cfg_scale == 4.0
    assert args.flag_relative_motion is True


def test_parse_args_honors_explicit_chunks_and_emit_frames() -> None:
    args = bench.parse_args([
        "--duration", "30",
        "--chunks", "2",
        "--emit-frames", "7",
        "--chunk-samples", "8000",
    ])

    assert args.chunks == 2
    assert args.emit_frames == 7
