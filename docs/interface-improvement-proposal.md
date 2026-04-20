# OmniRT Interface Improvement Proposal

This document proposes a practical interface roadmap for OmniRT from the user-experience perspective.

The current interface is good enough for internal engineering use and early adopters, but it is not yet complete or highly discoverable for broader open-source users.

## Current assessment

What works well today:

- one obvious top-level entrypoint: `omnirt generate`
- one unified request contract: `GenerateRequest`
- one unified result contract: `GenerateResult` plus `RunReport`
- backend, artifact export, and telemetry are already normalized across models

What is still weak for users:

- users must already know which `model` ids exist
- users cannot easily discover which parameters a model accepts
- `inputs` versus `config` is not self-explanatory
- schema validation is mostly runtime-only
- task coverage is narrower than the roadmap
- CLI is execution-oriented, not workflow-oriented

## Design goals

1. Keep OmniRT's public interface stable across model families.
2. Stay Diffusers-backed internally without forcing a raw Diffusers API on users.
3. Make the CLI and Python API self-discoverable.
4. Prefer explicit validation over runtime surprises.
5. Make future tasks such as `image2image`, `inpaint`, `edit`, and `video2video` fit naturally into the same contract family.

## Non-goals

- Replace Diffusers as the internal model execution layer.
- Expose every model's raw upstream argument surface directly to users.
- Make OmniRT fully Diffusers-compatible at the outermost API level.

## P0

These are the highest-value interface improvements and should come first.

### 1. Model discovery commands

Add:

- `omnirt models`
- `omnirt models <model-id>`

Expected behavior:

- list supported model ids
- show task, default backend, minimum memory hint, and current maturity
- show model-specific supported parameters and defaults
- show example commands for that model

Why this matters:

- it removes the biggest current usability gap
- it makes the CLI self-explanatory

### 2. First-class request validation

Add:

- `omnirt validate --config request.yaml`
- `omnirt generate --dry-run`

Expected behavior:

- validate task/model compatibility
- validate required inputs
- validate supported config keys
- print resolved defaults without executing generation

Why this matters:

- users can catch mistakes before a long or expensive run
- this is especially important for video models and remote model directories

### 3. Explicit parameter ownership rules

Document and enforce a simple rule:

- `inputs`: semantic generation inputs such as `prompt`, `negative_prompt`, `image`, `mask`, `control_image`
- `config`: execution settings such as `num_inference_steps`, `guidance_scale`, `height`, `width`, `dtype`, `seed`, `output_dir`

Why this matters:

- the current split is reasonable but not obvious
- a fixed rule makes requests easier to teach, validate, and extend

### 4. Better execution summaries

Improve CLI output after success:

- artifact path summary
- resolved model path
- resolved backend
- key generation settings
- a compact human-readable mode in addition to JSON

Why this matters:

- today the output is structurally useful but not optimized for quick scanning

### 5. Stronger user-facing errors

Errors should include:

- what was wrong
- what was expected
- a valid example
- suggested nearby models or tasks where possible

Example:

Instead of only saying a model does not support `text2image`, also suggest the supported task and a replacement command.

## P1

These improvements make the interface substantially more complete.

### 1. Typed per-task schemas

Introduce typed request variants internally, for example:

- `TextToImageRequest`
- `ImageToVideoRequest`
- `TextToVideoRequest`

These can still serialize into the current `GenerateRequest` envelope if desired.

Why this matters:

- better IDE help
- clearer validation
- less ambiguity around allowed keys

### 2. Model capability metadata

Extend the registry so each model exposes capability metadata such as:

- supported tasks
- required inputs
- optional inputs
- supported config keys
- scheduler support
- adapter support
- output artifact type

This should drive both validation and model help commands.

### 3. Presets

Add named presets such as:

- `fast`
- `balanced`
- `quality`
- `low-vram`

Why this matters:

- most users do not want to tune `guidance_scale`, `steps`, and `dtype` manually for every run

### 4. Fill out the missing task surfaces

The interface should grow to include:

- `image2image`
- `inpaint`
- `edit`
- `video2video`

These should not be bolted on ad hoc. They should fit the same task-plus-model contract cleanly.

### 5. Better Python ergonomics

Add helper constructors such as:

```python
from omnirt import requests

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic city at sunrise",
    width=1024,
    height=1024,
    guidance_scale=2.5,
)
```

Why this matters:

- the current dataclass API is serviceable but not very ergonomic

## P2

These are valuable, but not urgent.

### 1. Optional Diffusers-style convenience wrapper

Expose a convenience layer like:

```python
pipe = omnirt.pipeline("flux2.dev")
image = pipe(prompt="hello", num_inference_steps=30)
```

This should be optional sugar, not the primary API.

### 2. OpenAPI or service schema

If OmniRT is expected to back a service, define a stable service-facing schema and versioning plan.

### 3. Interactive CLI guidance

Longer term, add guided help such as:

- missing required input suggestions
- model recommendation by task
- preset recommendation based on available memory

## Proposed contract direction

The recommended direction is:

- keep the top-level `GenerateRequest` envelope
- make it strongly validated
- add model capability metadata
- add task-specific typed helpers
- keep model implementations free to adapt to Diffusers internally

This preserves OmniRT's runtime identity while fixing the biggest usability gaps.

## Suggested implementation order

1. Add model metadata to the registry.
2. Implement `omnirt models` and `omnirt models <id>`.
3. Implement `validate` and `--dry-run`.
4. Tighten schema validation for existing tasks.
5. Improve CLI success and error output.
6. Add typed task helpers in Python.
7. Expand the public task surface to editing and conversion workflows.

## Acceptance criteria

The interface should be considered meaningfully improved when a new user can:

1. discover supported models without reading source code
2. understand which arguments a model accepts without trial and error
3. validate a request before execution
4. read the CLI output and immediately know what happened
5. move from one model family to another without relearning the entire request shape

## Recommended first implementation slice

If only one short iteration is available, the best slice is:

- registry capability metadata
- `omnirt models`
- `omnirt validate`
- clearer errors

This would deliver the largest usability improvement for the least surface-area change.
