"""Command line interface for OmniRT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from typing import Optional, Sequence

from omnirt.api import describe_model, generate, list_available_models, validate
from omnirt.core.presets import available_presets
from omnirt.core.types import GenerateRequest, OmniRTError


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to a YAML or JSON request file.")
    parser.add_argument("--task", choices=["text2image", "text2video", "image2video"], help="Task to run.")
    parser.add_argument("--model", help="Model registry id to execute.")
    parser.add_argument("--backend", choices=["auto", "cuda", "ascend", "cpu-stub"], help="Override backend selection.")
    parser.add_argument("--prompt", help="Prompt for text2image, text2video, or prompt-guided image2video.")
    parser.add_argument("--negative-prompt", help="Negative prompt for prompt-driven generation.")
    parser.add_argument("--image", help="Input image for image2video generation.")
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
    parser.add_argument("--width", type=int, help="Output width for image generation.")
    parser.add_argument("--height", type=int, help="Output height for image generation.")
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
    parser.add_argument("--motion-bucket-id", type=int, help="Alias for SVD frame bucket / motion bucket id.")


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
    models_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")

    return parser


def request_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> GenerateRequest:
    if args.config:
        return GenerateRequest.from_file(args.config)

    if not args.task or not args.model:
        parser.error("either --config or both --task and --model are required")

    inputs = {}
    if args.task in {"text2image", "text2video"}:
        if not args.prompt:
            parser.error(f"--prompt is required for --task {args.task}")
        inputs["prompt"] = args.prompt
        if args.negative_prompt:
            inputs["negative_prompt"] = args.negative_prompt
        if args.task == "text2video":
            if args.num_frames is not None:
                inputs["num_frames"] = args.num_frames
            if args.fps is not None:
                inputs["fps"] = args.fps
    else:
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

    config = {}
    for field in (
        "num_inference_steps",
        "guidance_scale",
        "preset",
        "scheduler",
        "seed",
        "width",
        "height",
        "dtype",
        "num_images_per_prompt",
        "max_sequence_length",
        "caption_upsample_temperature",
        "output_dir",
        "frame_bucket",
        "motion_bucket_id",
        "decode_chunk_size",
        "noise_aug_strength",
    ):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.model_path:
        config["model_path"] = args.model_path

    return GenerateRequest(
        task=args.task,
        model=args.model,
        backend=args.backend or "auto",
        inputs=inputs,
        config=config,
    )


def render_model_summary(spec) -> str:
    caps = spec.capabilities
    lines = [
        f"model={spec.id}",
        f"task={spec.task}",
        f"default_backend={spec.default_backend}",
        f"maturity={caps.maturity}",
    ]
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
    if caps.supported_config:
        lines.append(f"supported_config={', '.join(caps.supported_config)}")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        if args.model:
            try:
                spec = describe_model(args.model)
            except (OmniRTError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            if args.json:
                print(
                    json.dumps(
                        {
                            "id": spec.id,
                            "task": spec.task,
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
                print(render_model_summary(spec))
            return 0

        specs = list_available_models(include_aliases=False)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "id": spec.id,
                            "task": spec.task,
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
                print(f"{spec.id}\t{spec.task}\t{spec.capabilities.maturity}\t{spec.capabilities.summary}")
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
