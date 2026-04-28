"""Command line interface for OmniRT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from omnirt.api import describe_model, generate, list_available_models, validate
from omnirt.backends import resolve_backend
from omnirt.bench import BenchScenario, get_bench_scenario, list_bench_scenarios, run_bench
from omnirt.core.presets import available_presets
from omnirt.core.registry import get_model, list_model_variants, supported_config_for_spec
from omnirt.core.types import GenerateRequest, OmniRTError
from omnirt.models import ensure_registered
from omnirt.models.flashtalk.resident_worker import FlashTalkResidentWorker
from omnirt.server import create_app
from omnirt.engine import GrpcWorkerServer, probe_worker_health
from omnirt.workers import ResidentWorkerService

PUBLIC_TASK_SURFACES = frozenset(
    {
        "text2image",
        "image2image",
        "text2video",
        "image2video",
        "audio2video",
        "text2audio",
    }
)


def task_surface_label(task: str) -> str:
    return "public" if task in PUBLIC_TASK_SURFACES else "preview"


def model_status_label(spec) -> str:
    return f"{task_surface_label(spec.task)}/{spec.capabilities.maturity}"


def render_supported_tasks(variants) -> str:
    return ", ".join(f"{task} ({model_status_label(spec)})" for task, spec in variants.items())


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to a YAML or JSON request file.")
    parser.add_argument(
        "--task",
        choices=["text2image", "image2image", "inpaint", "edit", "text2video", "image2video", "audio2video", "text2audio"],
        help="Task to run.",
    )
    parser.add_argument("--model", help="Model registry id to execute.")
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "ascend", "cpu-stub"],
        help="Override backend selection.",
    )
    parser.add_argument("--prompt", help="Prompt for image or video generation tasks.")
    parser.add_argument("--negative-prompt", help="Negative prompt for prompt-driven generation.")
    parser.add_argument("--reference-text", help="Reference transcript for text2audio voice cloning prompts.")
    parser.add_argument("--image", help="Input image for image-guided generation.")
    parser.add_argument("--mask", help="Input mask image for inpainting.")
    parser.add_argument("--audio", help="Input audio for audio2video generation.")
    parser.add_argument("--num-frames", type=int, help="Frame count for text2video or image2video generation.")
    parser.add_argument("--fps", type=int, help="Frames per second for exported video.")
    parser.add_argument("--frame-bucket", type=int, help="Motion bucket hint for SVD image2video.")
    parser.add_argument("--decode-chunk-size", type=int, help="Decode chunk size for video generation.")
    parser.add_argument("--noise-aug-strength", type=float, help="Noise augmentation for SVD image2video.")
    parser.add_argument("--num-inference-steps", type=int, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, help="Classifier-free guidance scale.")
    parser.add_argument("--preset", choices=available_presets(), help="Apply a named preset before explicit config values.")
    parser.add_argument("--scheduler", help="Scheduler override for models that support alternate schedulers.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--strength", type=float, help="Transformation strength for image editing tasks.")
    parser.add_argument("--width", type=int, help="Output width for image generation.")
    parser.add_argument("--height", type=int, help="Output height for image generation.")
    parser.add_argument("--size", help="Video size string for script-backed models, for example 416*720.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], help="Computation dtype.")
    parser.add_argument("--num-images-per-prompt", type=int, help="Images to generate per text-to-image prompt.")
    parser.add_argument("--max-sequence-length", type=int, help="Maximum prompt token length for Flux2.")
    parser.add_argument(
        "--caption-upsample-temperature",
        type=float,
        help="Caption upsample temperature for Flux2 caption expansion.",
    )
    parser.add_argument("--output-dir", help="Output directory for saved artifacts.")
    parser.add_argument("--model-path", help="Override the default model source.")
    parser.add_argument("--server-addr", help="Triton server address for external service-backed models.")
    parser.add_argument("--server-port", type=int, help="Triton gRPC server port for external service-backed models.")
    parser.add_argument("--sample-rate", type=int, help="Output audio sample rate for text2audio models.")
    parser.add_argument("--request-id", help="Stable external request id for deterministic service probes.")
    parser.add_argument("--motion-bucket-id", type=int, help="Alias for SVD frame bucket / motion bucket id.")
    parser.add_argument(
        "--repo-path",
        help="External repository checkout path for script-backed models such as SoulX-FlashTalk or SoulX-FlashHead.",
    )
    parser.add_argument("--ckpt-dir", help="Checkpoint directory for script-backed models.")
    parser.add_argument("--wav2vec-dir", help="wav2vec checkpoint directory for script-backed talking-head models.")
    parser.add_argument("--resident-target", help="Target gRPC address for a pre-warmed resident worker, for example 127.0.0.1:50071.")
    parser.add_argument("--resident-autostart", action="store_true", help="Auto-launch a managed resident worker for models that support it.")
    parser.add_argument("--audio-encode-mode", choices=["stream", "once"], help="Audio encoding mode for talking-head models.")
    parser.add_argument("--model-type", help="Model type for script-backed model families, for example FlashHead pro/lite.")
    parser.add_argument("--sample-steps", type=int, help="Sample step override for script-backed video/avatar models.")
    parser.add_argument(
        "--vae-2d-split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable FlashHead 2D VAE split.",
    )
    parser.add_argument("--latent-carry", action="store_true", help="Enable FlashHead latent carry experimental mode.")
    parser.add_argument("--npu-fusion-attention", action="store_true", help="Enable FlashHead NPU fusion attention.")
    parser.add_argument("--profile", action="store_true", help="Enable model-side profiling when supported.")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload for script-backed models that support it.")
    parser.add_argument("--max-chunks", type=int, help="Limit generated audio chunks for streaming avatar models.")
    parser.add_argument("--python-executable", help="Python interpreter used to launch external model scripts.")
    parser.add_argument(
        "--launcher",
        choices=["python", "torchrun", "accelerate"],
        help="External launcher for script-backed models.",
    )
    parser.add_argument("--nproc-per-node", type=int, help="Process count for multi-card torchrun launches.")
    parser.add_argument("--num-processes", type=int, help="Process count for accelerate launches.")
    parser.add_argument("--accelerate-executable", help="Accelerate executable used for launcher=accelerate.")
    parser.add_argument("--visible-devices", help="Device visibility override, for example 0,1,2,3,4,5,6,7 on Ascend.")
    parser.add_argument("--ascend-env-script", help="Ascend environment script to source before launching an external model.")
    parser.add_argument("--t5-quant", choices=["int8", "fp8"], help="T5 quantization mode for FlashTalk.")
    parser.add_argument("--t5-quant-dir", help="Directory containing T5 quantized weights for FlashTalk.")
    parser.add_argument("--wan-quant", choices=["int8", "fp8"], help="Wan weight-only quantization mode for FlashTalk.")
    parser.add_argument("--wan-quant-include", help="Comma-separated Wan module allowlist for FlashTalk quantization.")
    parser.add_argument("--wan-quant-exclude", help="Comma-separated Wan module denylist for FlashTalk quantization.")
    parser.add_argument("--t5-cpu", action="store_true", help="Place LiveAct T5 text encoder on CPU.")
    parser.add_argument("--rank0-t5-only", action="store_true", help="Only rank 0 performs LiveAct T5 encoding, sharing cached text context.")
    parser.add_argument("--stop-after-text-context", action="store_true", help="LiveAct debug mode: exit after text-context preparation.")
    parser.add_argument("--offload-cache", action="store_true", help="Offload LiveAct KV cache to CPU.")
    parser.add_argument("--fp8-kv-cache", action="store_true", help="Use FP8 LiveAct KV cache.")
    parser.add_argument("--block-offload", action="store_true", help="Offload LiveAct model blocks to CPU between forwards.")
    parser.add_argument("--audio-cfg", type=float, help="LiveAct audio classifier-free guidance scale.")
    parser.add_argument("--dura-print", action="store_true", help="Print LiveAct per-block duration details.")
    parser.add_argument("--steam-audio", action="store_true", help="Use LiveAct streaming-audio path; name follows upstream's steam_audio flag.")
    parser.add_argument("--mean-memory", action="store_true", help="Enable LiveAct mean-memory strategy.")
    parser.add_argument("--use-cache-vae", action="store_true", help="Use LiveAct cached VAE decode.")
    parser.add_argument("--vae-path", help="LiveAct VAE checkpoint override, for example models/vae/lightvaew2_1.pth.")
    parser.add_argument("--use-lightvae", action="store_true", help="Use LiveAct LightVAE architecture.")
    parser.add_argument("--condition-cache-dir", help="LiveAct visual condition cache directory.")
    parser.add_argument("--disable-condition-cache", action="store_true", help="Disable LiveAct visual condition cache.")
    parser.add_argument("--text-cache-device", help="Device label for LiveAct prepare_text_cache.py, for example npu.")
    parser.add_argument("--text-cache-visible-devices", help="Single-device visibility override for LiveAct text-cache preparation.")
    parser.add_argument("--force-text-cache", action="store_true", help="Regenerate LiveAct text cache even when the cache file exists.")
    parser.add_argument("--disable-text-cache-prepare", action="store_true", help="Skip LiveAct prepare_text_cache.py before generation.")
    parser.add_argument("--fast-export", action="store_true", help="Enable LiveAct experimental fast ffmpeg export.")
    parser.add_argument("--disable-fast-export", action="store_true", help="Force LiveAct legacy export path.")
    parser.add_argument("--fast-export-preset", help="LiveAct fast-export libx264 preset.")
    parser.add_argument("--fast-export-crf", type=int, help="LiveAct fast-export CRF.")
    parser.add_argument("--sequence-parallel-degree", type=int, help="LiveAct sequence parallel degree.")
    parser.add_argument("--ulysses-degree", type=int, help="LiveAct Ulysses degree.")
    parser.add_argument("--ring-degree", type=int, help="LiveAct ring degree.")
    parser.add_argument("--stage-profile", action="store_true", help="Enable LiveAct stage-profile aggregation.")
    parser.add_argument("--device-map", help="Diffusers device placement policy such as balanced or unet:0,vae:1.")
    parser.add_argument("--devices", help="Comma-separated device list used to infer balanced placement, for example cuda:0,cuda:1.")
    parser.add_argument("--cache", choices=["tea_cache"], help="Enable cache middleware, currently tea_cache.")
    parser.add_argument("--quantization", choices=["int8", "fp8", "nf4"], help="Best-effort runtime quantization mode.")
    parser.add_argument(
        "--quantization-backend",
        choices=["auto", "torchao", "bitsandbytes"],
        help="Implementation hint for runtime quantization.",
    )
    parser.add_argument("--enable-layerwise-casting", action="store_true", help="Enable best-effort layerwise casting hooks.")
    parser.add_argument("--layerwise-casting-storage-dtype", help="Storage dtype hint for layerwise casting.")
    parser.add_argument("--layerwise-casting-compute-dtype", help="Compute dtype hint for layerwise casting.")
    parser.add_argument("--enable-tea-cache", action="store_true", help="Enable best-effort TeaCache hooks when supported.")
    parser.add_argument("--tea-cache-ratio", type=float, help="TeaCache reuse ratio hint.")
    parser.add_argument("--tea-cache-interval", type=int, help="TeaCache interval hint.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnirt", description="OmniRT command line interface.")
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Run a generation request.")
    add_request_arguments(generate_parser)
    generate_parser.add_argument("--dry-run", action="store_true", help="Validate and resolve defaults without executing.")
    generate_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")

    validate_parser = subparsers.add_parser("validate", help="Validate a generation request without executing.")
    add_request_arguments(validate_parser)
    validate_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")

    models_parser = subparsers.add_parser("models", help="List supported models or show one model in detail.")
    models_parser.add_argument("model", nargs="?", help="Optional model id to describe.")
    output_group = models_parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")
    output_group.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format for the list view. Markdown is deterministic and suitable for docs generation.",
    )

    serve_parser = subparsers.add_parser("serve", help="Run the OmniRT HTTP API server.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    serve_parser.add_argument("--port", type=int, default=8000, help="TCP port to bind.")
    serve_parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "ascend", "cpu-stub"],
        default="auto",
        help="Default backend for requests that omit backend.",
    )
    serve_parser.add_argument("--max-concurrency", type=int, default=1, help="Worker concurrency for queued jobs.")
    serve_parser.add_argument("--pipeline-cache-size", type=int, default=4, help="Maximum cached executor instances.")
    serve_parser.add_argument("--api-key-file", help="Optional newline-delimited API key file.")
    serve_parser.add_argument("--model-aliases", help="Optional YAML/JSON alias mapping file.")
    serve_parser.add_argument("--redis-url", help="Optional Redis URL for cross-process job storage.")
    serve_parser.add_argument("--otlp-endpoint", help="Optional OTLP/HTTP endpoint used to export traces.")
    serve_parser.add_argument(
        "--remote-worker",
        action="append",
        default=[],
        help="Remote worker spec: worker_id=host:port@model1,model2#tag1,tag2 . May be repeated.",
    )
    serve_parser.add_argument("--device-map", help="Default device placement policy for incoming requests.")
    serve_parser.add_argument("--devices", help="Default comma-separated device list for incoming requests.")
    serve_parser.add_argument("--batch-window-ms", type=int, default=0, help="Queue batching window in milliseconds.")
    serve_parser.add_argument("--max-batch-size", type=int, default=1, help="Maximum requests merged into one batch.")

    bench_parser = subparsers.add_parser("bench", help="Run a local benchmark scenario.")
    add_request_arguments(bench_parser)
    bench_parser.add_argument("--scenario", choices=list_bench_scenarios(), help="Built-in benchmark scenario.")
    bench_parser.add_argument("--concurrency", type=int, help="Concurrent request count.")
    bench_parser.add_argument("--total", type=int, default=10, help="Total measured requests.")
    bench_parser.add_argument("--warmup", type=int, default=1, help="Warmup requests before timing.")
    bench_parser.add_argument("--batch-window-ms", type=int, default=0, help="Queue batching window in milliseconds.")
    bench_parser.add_argument("--max-batch-size", type=int, default=1, help="Maximum requests merged into one batch.")
    bench_parser.add_argument("--output", help="Optional JSON output path.")
    bench_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")

    worker_parser = subparsers.add_parser("worker", help="Run the OmniRT gRPC worker server.")
    worker_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    worker_parser.add_argument("--port", type=int, default=50061, help="TCP port to bind.")
    worker_parser.add_argument("--worker-id", default="worker", help="Stable worker identifier.")
    worker_parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "ascend", "cpu-stub"],
        default="auto",
        help="Default backend for requests that omit backend.",
    )
    worker_parser.add_argument("--max-concurrency", type=int, default=1, help="Worker concurrency for execution.")
    worker_parser.add_argument("--pipeline-cache-size", type=int, default=4, help="Maximum cached executor instances.")
    worker_parser.add_argument("--redis-url", help="Optional Redis URL for cross-process job storage.")
    worker_parser.add_argument("--otlp-endpoint", help="Optional OTLP/HTTP endpoint used to export traces.")

    resident_flashtalk_parser = subparsers.add_parser(
        "resident-flashtalk-worker",
        help="Run a pre-warmed gRPC resident worker for SoulX-FlashTalk.",
    )
    resident_flashtalk_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    resident_flashtalk_parser.add_argument("--port", type=int, default=50071, help="TCP port to bind.")
    resident_flashtalk_parser.add_argument("--worker-id", default="flashtalk-resident", help="Stable worker identifier.")
    resident_flashtalk_parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "ascend", "cpu-stub"],
        default="auto",
        help="Backend used to initialize the resident worker.",
    )
    resident_flashtalk_parser.add_argument("--repo-path", help="External SoulX-FlashTalk checkout path.")
    resident_flashtalk_parser.add_argument("--ckpt-dir", help="Checkpoint directory for SoulX-FlashTalk.")
    resident_flashtalk_parser.add_argument("--wav2vec-dir", help="wav2vec checkpoint directory for FlashTalk.")
    resident_flashtalk_parser.add_argument("--audio-encode-mode", choices=["stream", "once"], help="Default audio encoding mode.")
    resident_flashtalk_parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload for the resident worker.")
    resident_flashtalk_parser.add_argument("--max-chunks", type=int, help="Default chunk cap for streaming requests.")
    resident_flashtalk_parser.add_argument("--python-executable", help="Python interpreter used by FlashTalk helper code.")
    resident_flashtalk_parser.add_argument(
        "--launcher",
        choices=["python", "torchrun", "accelerate"],
        help="Launcher configuration carried into the resident worker.",
    )
    resident_flashtalk_parser.add_argument("--nproc-per-node", type=int, help="Process count for multi-card torchrun launches.")
    resident_flashtalk_parser.add_argument("--num-processes", type=int, help="Process count for accelerate launches.")
    resident_flashtalk_parser.add_argument("--accelerate-executable", help="Accelerate executable used for launcher=accelerate.")
    resident_flashtalk_parser.add_argument("--visible-devices", help="Visible devices override, for example 0,1,2,3,4,5,6,7.")
    resident_flashtalk_parser.add_argument("--ascend-env-script", help="Ascend environment script path.")
    resident_flashtalk_parser.add_argument("--t5-quant", choices=["int8", "fp8"], help="T5 quantization mode.")
    resident_flashtalk_parser.add_argument("--t5-quant-dir", help="Directory containing T5 quantized weights.")
    resident_flashtalk_parser.add_argument("--wan-quant", choices=["int8", "fp8"], help="Wan quantization mode.")
    resident_flashtalk_parser.add_argument("--wan-quant-include", help="Comma-separated Wan module allowlist.")
    resident_flashtalk_parser.add_argument("--wan-quant-exclude", help="Comma-separated Wan module denylist.")
    resident_flashtalk_parser.add_argument("--output-dir", help="Default output directory for resident-worker requests.")
    resident_flashtalk_parser.add_argument("--seed", type=int, help="Default seed when the request omits one.")

    return parser


def parse_remote_worker_specs(specs: Sequence[str]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for spec in specs:
        worker_id_and_target, sep, remainder = spec.partition("=")
        if not sep or not worker_id_and_target or not remainder:
            raise ValueError(f"Invalid remote worker spec {spec!r}; expected worker_id=host:port@models")
        address_part, at_sep, models_and_tags = remainder.partition("@")
        models_part, hash_sep, tags_part = models_and_tags.partition("#")
        models = tuple(item for item in models_part.split(",") if item) if at_sep else ()
        tags = tuple(item for item in tags_part.split(",") if item) if hash_sep else ()
        parsed.append(
            {
                "worker_id": worker_id_and_target,
                "address": address_part,
                "models": models,
                "tags": tags,
            }
        )
    return parsed


def flashtalk_worker_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for field in (
        "repo_path",
        "ckpt_dir",
        "wav2vec_dir",
        "audio_encode_mode",
        "max_chunks",
        "python_executable",
        "launcher",
        "nproc_per_node",
        "num_processes",
        "accelerate_executable",
        "visible_devices",
        "ascend_env_script",
        "t5_quant",
        "t5_quant_dir",
        "wan_quant",
        "wan_quant_include",
        "wan_quant_exclude",
        "output_dir",
        "seed",
    ):
        value = getattr(args, field, None)
        if value is not None:
            config[field] = value
    if getattr(args, "cpu_offload", False):
        config["cpu_offload"] = True
    if getattr(args, "resident_autostart", False):
        config["resident_autostart"] = True
    return config


def request_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> GenerateRequest:
    if args.config:
        return GenerateRequest.from_file(args.config)

    if not args.task or not args.model:
        parser.error("either --config or both --task and --model are required")

    inputs = {}
    if args.task in {"text2image", "text2video", "text2audio"}:
        if not args.prompt:
            parser.error(f"--prompt is required for --task {args.task}")
        inputs["prompt"] = args.prompt
        if args.negative_prompt:
            inputs["negative_prompt"] = args.negative_prompt
        if args.task == "text2audio":
            if not args.audio:
                parser.error("--audio is required for --task text2audio")
            inputs["audio"] = args.audio
            if args.reference_text:
                inputs["reference_text"] = args.reference_text
        if args.task == "text2video":
            if args.num_frames is not None:
                inputs["num_frames"] = args.num_frames
            if args.fps is not None:
                inputs["fps"] = args.fps
    elif args.task == "image2video":
        if not args.image:
            parser.error("--image is required for --task image2video")
        inputs["image"] = args.image
        if args.prompt:
            inputs["prompt"] = args.prompt
        if args.negative_prompt:
            inputs["negative_prompt"] = args.negative_prompt
        if args.num_frames is not None:
            inputs["num_frames"] = args.num_frames
        if args.fps is not None:
            inputs["fps"] = args.fps
    elif args.task == "image2image":
        if not args.image:
            parser.error("--image is required for --task image2image")
        if not args.prompt:
            parser.error("--prompt is required for --task image2image")
        inputs["image"] = args.image
        inputs["prompt"] = args.prompt
        if args.negative_prompt:
            inputs["negative_prompt"] = args.negative_prompt
    elif args.task == "inpaint":
        if not args.image:
            parser.error("--image is required for --task inpaint")
        if not args.mask:
            parser.error("--mask is required for --task inpaint")
        if not args.prompt:
            parser.error("--prompt is required for --task inpaint")
        inputs["image"] = args.image
        inputs["mask"] = args.mask
        inputs["prompt"] = args.prompt
        if args.negative_prompt:
            inputs["negative_prompt"] = args.negative_prompt
    elif args.task == "edit":
        if not args.image:
            parser.error("--image is required for --task edit")
        if not args.prompt:
            parser.error("--prompt is required for --task edit")
        inputs["image"] = args.image
        inputs["prompt"] = args.prompt
    else:
        if not args.image:
            parser.error("--image is required for --task audio2video")
        if not args.audio:
            parser.error("--audio is required for --task audio2video")
        inputs["image"] = args.image
        inputs["audio"] = args.audio
        if args.prompt:
            inputs["prompt"] = args.prompt

    config = {}
    for field in (
        "num_inference_steps",
        "guidance_scale",
        "preset",
        "scheduler",
        "seed",
        "strength",
        "width",
        "height",
        "size",
        "fps",
        "dtype",
        "num_images_per_prompt",
        "max_sequence_length",
        "caption_upsample_temperature",
        "output_dir",
        "server_addr",
        "server_port",
        "sample_rate",
        "request_id",
        "frame_bucket",
        "motion_bucket_id",
        "decode_chunk_size",
        "noise_aug_strength",
        "repo_path",
        "ckpt_dir",
        "wav2vec_dir",
        "resident_target",
        "audio_encode_mode",
        "model_type",
        "sample_steps",
        "vae_2d_split",
        "max_chunks",
        "python_executable",
        "launcher",
        "nproc_per_node",
        "num_processes",
        "accelerate_executable",
        "visible_devices",
        "ascend_env_script",
        "t5_quant",
        "t5_quant_dir",
        "wan_quant",
        "wan_quant_include",
        "wan_quant_exclude",
        "audio_cfg",
        "vae_path",
        "condition_cache_dir",
        "text_cache_device",
        "text_cache_visible_devices",
        "fast_export_preset",
        "fast_export_crf",
        "sequence_parallel_degree",
        "ulysses_degree",
        "ring_degree",
        "device_map",
        "devices",
        "cache",
        "quantization",
        "quantization_backend",
        "layerwise_casting_storage_dtype",
        "layerwise_casting_compute_dtype",
        "tea_cache_ratio",
        "tea_cache_interval",
    ):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.model_path:
        config["model_path"] = args.model_path
    if getattr(args, "resident_autostart", False):
        config["resident_autostart"] = True
    if getattr(args, "cpu_offload", False):
        config["cpu_offload"] = True
    for flag in (
        "t5_cpu",
        "rank0_t5_only",
        "stop_after_text_context",
        "offload_cache",
        "fp8_kv_cache",
        "block_offload",
        "dura_print",
        "steam_audio",
        "mean_memory",
        "use_cache_vae",
        "use_lightvae",
        "disable_condition_cache",
        "force_text_cache",
        "disable_text_cache_prepare",
        "fast_export",
        "disable_fast_export",
        "stage_profile",
        "latent_carry",
        "npu_fusion_attention",
        "profile",
    ):
        if getattr(args, flag, False):
            config[flag] = True
    if getattr(args, "enable_layerwise_casting", False):
        config["enable_layerwise_casting"] = True
    if getattr(args, "enable_tea_cache", False):
        config["enable_tea_cache"] = True

    return GenerateRequest(
        task=args.task,
        model=args.model,
        backend=args.backend or "auto",
        inputs=inputs,
        config=config,
    )


def render_model_summary(spec, *, variants=None) -> str:
    caps = spec.capabilities
    lines = [
        f"model={spec.id}",
        f"task={spec.task}",
        f"status={model_status_label(spec)}",
        f"default_backend={spec.default_backend}",
        f"maturity={caps.maturity}",
    ]
    if variants:
        lines.append(f"supported_tasks={render_supported_tasks(variants)}")
    if caps.summary:
        lines.append(f"summary={caps.summary}")
    if caps.alias_of:
        lines.append(f"alias_of={caps.alias_of}")
    if spec.resource_hint:
        lines.append(f"resource_hint={json.dumps(spec.resource_hint, ensure_ascii=False, sort_keys=True)}")
    if caps.required_inputs:
        lines.append(f"required_inputs={', '.join(caps.required_inputs)}")
    if caps.optional_inputs:
        lines.append(f"optional_inputs={', '.join(caps.optional_inputs)}")
    supported_config = supported_config_for_spec(spec)
    if supported_config:
        lines.append(f"supported_config={', '.join(supported_config)}")
    if caps.default_config:
        lines.append(f"default_config={json.dumps(caps.default_config, ensure_ascii=False, sort_keys=True)}")
    if caps.supported_schedulers:
        lines.append(f"supported_schedulers={', '.join(caps.supported_schedulers)}")
    lines.append(f"presets={', '.join(available_presets())}")
    if caps.adapter_kinds:
        lines.append(f"adapter_kinds={', '.join(caps.adapter_kinds)}")
    if caps.artifact_kind:
        lines.append(f"artifact_kind={caps.artifact_kind}")
    if caps.example:
        lines.append(f"example={caps.example}")
    return "\n".join(lines)


_MARKDOWN_TASK_ORDER = ("text2image", "text2audio", "text2video", "image2video", "audio2video", "image2image", "inpaint", "edit")
_MARKDOWN_TASK_HEADINGS = {
    "text2image": "Text to image",
    "text2audio": "Text to audio",
    "text2video": "Text to video",
    "image2video": "Image to video",
    "audio2video": "Audio to video",
    "image2image": "Image to image",
    "inpaint": "Inpainting",
    "edit": "Image editing",
}


def render_models_markdown(specs) -> str:
    """Render the full model registry as a deterministic markdown document.

    The output groups registry ids by primary task, then lists any alias
    relationships declared via ``ModelCapabilities.alias_of`` so docs can
    surface that ``flux2.dev`` and ``flux2-dev`` are the same pipeline.
    """

    by_task: dict = {}
    aliases: list = []
    seen_ids: set = set()
    for spec in sorted(specs, key=lambda s: s.id):
        if spec.id in seen_ids:
            continue
        seen_ids.add(spec.id)
        if spec.capabilities.alias_of:
            aliases.append((spec.id, spec.capabilities.alias_of))
            continue
        by_task.setdefault(spec.task, []).append(spec)

    lines = ["# OmniRT supported models", "", "<!-- GENERATED by scripts/generate_models_doc.py — do not edit -->", ""]

    ordered_tasks = [t for t in _MARKDOWN_TASK_ORDER if t in by_task]
    ordered_tasks += sorted(set(by_task) - set(_MARKDOWN_TASK_ORDER))
    for task in ordered_tasks:
        heading = _MARKDOWN_TASK_HEADINGS.get(task, task)
        lines.append(f"## {heading}")
        lines.append("")
        lines.append("| Registry id | Maturity | Summary |")
        lines.append("|---|---|---|")
        for spec in sorted(by_task[task], key=lambda s: s.id):
            summary = spec.capabilities.summary.replace("|", "\\|") if spec.capabilities.summary else ""
            lines.append(f"| `{spec.id}` | {spec.capabilities.maturity} | {summary} |")
        lines.append("")

    if aliases:
        lines.append("## Aliases")
        lines.append("")
        lines.append("| Alias | Canonical id |")
        lines.append("|---|---|")
        for alias_id, canonical_id in sorted(aliases):
            lines.append(f"| `{alias_id}` | `{canonical_id}` |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_validation_summary(validation) -> str:
    lines = [
        f"ok={str(validation.ok).lower()}",
        f"task={validation.request.task}",
        f"model={validation.request.model}",
    ]
    if validation.resolved_backend:
        lines.append(f"resolved_backend={validation.resolved_backend}")
    if validation.resolved_inputs:
        lines.append(f"resolved_inputs={json.dumps(validation.resolved_inputs, ensure_ascii=False, sort_keys=True)}")
    if validation.resolved_config:
        lines.append(f"resolved_config={json.dumps(validation.resolved_config, ensure_ascii=False, sort_keys=True)}")
    for issue in validation.issues:
        lines.append(f"{issue.level}: {issue.message}")
    return "\n".join(lines)


def render_generate_summary(payload: dict) -> str:
    metadata = payload.get("metadata", {})
    outputs = payload.get("outputs", [])
    lines = [
        f"run_id={metadata.get('run_id', '')}",
        f"model={metadata.get('model', '')}",
        f"task={metadata.get('task', '')}",
        f"backend={metadata.get('backend', '')}",
    ]
    resolved = metadata.get("config_resolved", {})
    for key in ("model_path", "scheduler", "height", "width", "num_frames", "fps", "num_inference_steps", "guidance_scale", "seed"):
        if key in resolved:
            lines.append(f"{key}={resolved[key]}")
    if outputs:
        lines.append(f"artifacts={len(outputs)}")
        for output in outputs:
            lines.append(f"artifact: {output.get('path', '')} ({output.get('mime', '')})")
    return "\n".join(lines)


def render_bench_summary(payload: dict) -> str:
    latency = payload.get("latency_ms", {})
    ttft = payload.get("ttft_ms", {})
    return "\n".join(
        [
            f"scenario={payload.get('scenario', '')}",
            f"total_requests={payload.get('total_requests', 0)}",
            f"concurrency={payload.get('concurrency', 0)}",
            f"throughput_rps={payload.get('throughput_rps', 0)}",
            f"latency_p50_ms={latency.get('p50', 0)}",
            f"latency_p95_ms={latency.get('p95', 0)}",
            f"latency_p99_ms={latency.get('p99', 0)}",
            f"ttft_p50_ms={ttft.get('p50', 0)}",
            f"peak_vram={payload.get('peak_vram', 0)}",
            f"cache_hit_ratio={payload.get('cache_hit_ratio', 0)}",
            f"batch_size_mean={payload.get('batch_size_mean', 0)}",
            f"batched_request_ratio={payload.get('batched_request_ratio', 0)}",
        ]
    )


def scenario_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> BenchScenario:
    if args.scenario:
        base = get_bench_scenario(args.scenario)
        return BenchScenario(
            name=base.name,
            request_template=base.request_template,
            concurrency=args.concurrency or base.concurrency,
            total_requests=args.total or base.total_requests,
            warmup=args.warmup,
            batch_window_ms=args.batch_window_ms if args.batch_window_ms is not None else base.batch_window_ms,
            max_batch_size=args.max_batch_size if args.max_batch_size is not None else base.max_batch_size,
        )

    request = request_from_args(args, parser)
    return BenchScenario(
        name=f"{request.model}-{request.task}",
        request_template=request,
        concurrency=args.concurrency or 1,
        total_requests=args.total,
        warmup=args.warmup,
        batch_window_ms=args.batch_window_ms,
        max_batch_size=args.max_batch_size,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            print(
                "error: uvicorn is required for `omnirt serve`. Install `omnirt[server]` extras first.",
                file=sys.stderr,
            )
            return 2

        try:
            remote_workers = parse_remote_worker_specs(args.remote_worker or [])
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for worker in remote_workers:
            try:
                probe_worker_health(str(worker["address"]))
            except Exception as exc:
                print(f"error: remote worker {worker['worker_id']} is unreachable: {exc}", file=sys.stderr)
                return 2
        app = create_app(
            default_backend=args.backend,
            max_concurrency=args.max_concurrency,
            pipeline_cache_size=args.pipeline_cache_size,
            default_request_config={
                key: value
                for key, value in {
                    "device_map": args.device_map,
                    "devices": args.devices,
                }.items()
                if value is not None
            },
            api_key_file=args.api_key_file,
            model_aliases_path=args.model_aliases,
            batch_window_ms=args.batch_window_ms,
            max_batch_size=args.max_batch_size,
            redis_url=args.redis_url,
            otlp_endpoint=args.otlp_endpoint,
            remote_workers=remote_workers,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "worker":
        from omnirt.engine import OmniEngine
        from omnirt.engine.redis_store import RedisJobStore
        from omnirt.telemetry import OtlpExporter, TraceRecorder

        tracer = TraceRecorder(exporters=[OtlpExporter(endpoint=args.otlp_endpoint)] if args.otlp_endpoint else None)
        job_store = RedisJobStore(redis_url=args.redis_url) if args.redis_url else None

        worker_engine = OmniEngine(
            max_concurrency=args.max_concurrency,
            pipeline_cache_size=args.pipeline_cache_size,
            worker_id=args.worker_id,
            tracer=tracer,
            job_store=job_store,
        )
        server = GrpcWorkerServer(worker_engine, host=args.host, port=args.port).start()
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            server.stop(0.0)
        return 0

    if args.command == "resident-flashtalk-worker":
        ensure_registered()
        worker_runtime = resolve_backend(args.backend)
        model_spec = get_model("soulx-flashtalk-14b", task="audio2video")
        worker = FlashTalkResidentWorker(
            runtime=worker_runtime,
            model_spec=model_spec,
            config=flashtalk_worker_config_from_args(args),
            adapters=None,
        )
        service = ResidentWorkerService(worker, worker_id=args.worker_id)
        service.handle.start()
        if not getattr(worker, "serves_rpc", True):
            try:
                worker.serve_forever()
            except KeyboardInterrupt:
                worker.shutdown()
            return 0
        server = GrpcWorkerServer(service, host=args.host, port=args.port).start()
        try:
            worker.serve_forever()
        except KeyboardInterrupt:
            server.stop(0.0)
        finally:
            server.stop(0.0)
            worker.shutdown()
        return 0

    if args.command == "bench":
        try:
            scenario = scenario_from_args(args, parser)
            report = run_bench(scenario)
        except (OmniRTError, ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        payload = report.to_dict()
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        else:
            print(render_bench_summary(payload))
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "models":
        if args.model:
            try:
                spec = describe_model(args.model)
            except (OmniRTError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            variants = list_model_variants(args.model)
            if args.json:
                print(
                    json.dumps(
                        {
                            "id": spec.id,
                            "task": spec.task,
                            "status": model_status_label(spec),
                            "supported_tasks": list(variants),
                            "supported_task_statuses": {
                                task: model_status_label(variant_spec) for task, variant_spec in variants.items()
                            },
                            "default_backend": spec.default_backend,
                            "resource_hint": spec.resource_hint,
                            "presets": list(available_presets()),
                            "capabilities": asdict(spec.capabilities),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(render_model_summary(spec, variants=variants))
            return 0

        include_aliases = getattr(args, "format", "text") == "markdown"
        specs = list_available_models(include_aliases=include_aliases)
        if getattr(args, "format", "text") == "markdown":
            sys.stdout.write(render_models_markdown(specs))
            return 0
        if args.json or getattr(args, "format", "text") == "json":
            print(
                json.dumps(
                    [
                        {
                            "id": spec.id,
                            "task": spec.task,
                            "status": model_status_label(spec),
                            "default_backend": spec.default_backend,
                            "maturity": spec.capabilities.maturity,
                            "summary": spec.capabilities.summary,
                            "presets": list(available_presets()),
                        }
                        for spec in specs
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            for spec in specs:
                print(f"{spec.id}\t{spec.task}\t{model_status_label(spec)}\t{spec.capabilities.summary}")
        return 0

    if args.command not in {"generate", "validate"}:
        parser.print_help()
        return 0

    request = request_from_args(args, parser)
    if args.command == "validate":
        try:
            validation = validate(request, backend=args.backend)
        except (OmniRTError, ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(validation.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(render_validation_summary(validation))
        return 0 if validation.ok else 2

    if getattr(args, "dry_run", False):
        try:
            validation = validate(request, backend=args.backend)
        except (OmniRTError, ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(validation.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(render_validation_summary(validation))
        return 0 if validation.ok else 2

    try:
        result = generate(request, backend=args.backend)
    except (OmniRTError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    else:
        print(render_generate_summary(payload))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0
