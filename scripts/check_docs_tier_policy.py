#!/usr/bin/env python3
"""Guard docs against promoting experimental models as default examples."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Rule:
    path: str
    pattern: re.Pattern[str]
    message: str


RULES: tuple[Rule, ...] = (
    Rule(
        path="docs/getting_started/quickstart.md",
        pattern=re.compile(r"--model\s+sd15\b|model:\s*sd15\b"),
        message="quickstart generation examples must use core/adjacent models",
    ),
    Rule(
        path="docs/getting_started/quickstart.en.md",
        pattern=re.compile(r"--model\s+sd15\b|model:\s*sd15\b"),
        message="quickstart generation examples must use core/adjacent models",
    ),
    Rule(
        path="docs/index.md",
        pattern=re.compile(r"`(?:sd15|sd21|cogvideox-[^`]+|hunyuan-video[^`]*)`"),
        message="front-page representative model tables must not promote experimental models",
    ),
    Rule(
        path="docs/index.en.md",
        pattern=re.compile(r"`(?:sd15|sd21|cogvideox-[^`]+|hunyuan-video[^`]*)`"),
        message="front-page representative model tables must not promote experimental models",
    ),
    Rule(
        path="docs/user_guide/generation/index.md",
        pattern=re.compile(r"`(?:sd15|sd21|cogvideox-[^`]+|hunyuan-video[^`]*)`"),
        message="generation overview tables must default to core/adjacent models",
    ),
    Rule(
        path="docs/user_guide/generation/index.en.md",
        pattern=re.compile(r"`(?:sd15|sd21|cogvideox-[^`]+|hunyuan-video[^`]*)`"),
        message="generation overview tables must default to core/adjacent models",
    ),
    Rule(
        path="docs/user_guide/generation/text2image.md",
        pattern=re.compile(r'(--model\s+sd15|model="sd15"|"model":\s*"sd15")'),
        message="text2image default examples must not use experimental SD1.5",
    ),
    Rule(
        path="docs/user_guide/generation/text2image.en.md",
        pattern=re.compile(r'(--model\s+sd15|model="sd15"|"model":\s*"sd15")'),
        message="text2image default examples must not use experimental SD1.5",
    ),
    Rule(
        path="docs/user_guide/generation/text2video.md",
        pattern=re.compile(r'(--model\s+(?:cogvideox|hunyuan)|model="(?:cogvideox|hunyuan)|"model":\s*"(?:cogvideox|hunyuan))'),
        message="text2video recipes must not use experimental video families as defaults",
    ),
    Rule(
        path="docs/user_guide/generation/text2video.en.md",
        pattern=re.compile(r'(--model\s+(?:cogvideox|hunyuan)|model="(?:cogvideox|hunyuan)|"model":\s*"(?:cogvideox|hunyuan))'),
        message="text2video recipes must not use experimental video families as defaults",
    ),
    Rule(
        path="docs/user_guide/features/validation.md",
        pattern=re.compile(r'(--model\s+sd15|model="sd15"|"model":\s*"sd15")'),
        message="validation examples must use core/adjacent models",
    ),
    Rule(
        path="docs/user_guide/features/validation.en.md",
        pattern=re.compile(r'(--model\s+sd15|model="sd15"|"model":\s*"sd15")'),
        message="validation examples must use core/adjacent models",
    ),
    Rule(
        path="docs/user_guide/serving/cli.md",
        pattern=re.compile(r"--model\s+sd15\b"),
        message="serving CLI examples must use core/adjacent models",
    ),
    Rule(
        path="docs/user_guide/serving/cli.en.md",
        pattern=re.compile(r"--model\s+sd15\b"),
        message="serving CLI examples must use core/adjacent models",
    ),
    Rule(
        path="docs/user_guide/serving/python_api.md",
        pattern=re.compile(r'pipeline\("sd15"'),
        message="pipeline examples must use core/adjacent models",
    ),
    Rule(
        path="docs/user_guide/serving/python_api.en.md",
        pattern=re.compile(r'pipeline\("sd15"'),
        message="pipeline examples must use core/adjacent models",
    ),
    Rule(
        path="docs/developer_guide/backend_onboarding.md",
        pattern=re.compile(r"smoke test.*`sd15`|smoke test.*SD1\.5|smoke test.*SD 1\.5", re.IGNORECASE),
        message="backend onboarding smoke guidance must use core/adjacent baselines",
    ),
    Rule(
        path="docs/developer_guide/backend_onboarding.en.md",
        pattern=re.compile(r"smoke test.*`sd15`|smoke test.*SD1\.5|smoke test.*SD 1\.5", re.IGNORECASE),
        message="backend onboarding smoke guidance must use core/adjacent baselines",
    ),
)


def main() -> int:
    failures: list[str] = []
    for rule in RULES:
        path = REPO_ROOT / rule.path
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rule.pattern.search(line):
                failures.append(f"{rule.path}:{lineno}: {rule.message}: {line.strip()}")

    if failures:
        print("docs tier policy check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("docs tier policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
