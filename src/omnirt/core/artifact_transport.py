"""Inline-bytes artifact transport.

Two functions here — one the worker runs, one the client runs. The shape of
:class:`omnirt.core.types.Artifact` is intentionally minimal so the wire
format stays stable: ``transport="path"`` means the ``path`` field is
authoritative (client and worker share a filesystem); ``transport="inline_bytes"``
means ``data_b64`` carries the file contents and ``path`` is the *original*
filename on the worker (preserved so clients can derive an extension or log
the source).

The cap is governed by ``OMNIRT_ARTIFACT_INLINE_MAX_MB`` (default 64). When a
worker tries to inline an artifact that exceeds the cap, we raise
:class:`ArtifactTooLargeError` rather than silently falling back to ``path``,
because the silent-fallback behavior is exactly what made the original
single-box deployment drift into a broken cross-host deployment without
anyone noticing. Make it loud.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from omnirt.core.types import Artifact, ArtifactTooLargeError


_DEFAULT_INLINE_MAX_MB = 64
_ENV_INLINE_MAX_MB = "OMNIRT_ARTIFACT_INLINE_MAX_MB"


def inline_limit_bytes() -> int:
    """Resolve the inline-bytes budget from env, in bytes."""
    raw = os.environ.get(_ENV_INLINE_MAX_MB, "").strip()
    if not raw:
        value_mb = _DEFAULT_INLINE_MAX_MB
    else:
        try:
            value_mb = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{_ENV_INLINE_MAX_MB} must be an integer number of megabytes; got {raw!r}"
            ) from exc
        if value_mb <= 0:
            raise ValueError(f"{_ENV_INLINE_MAX_MB} must be positive; got {value_mb}")
    return value_mb * 1024 * 1024


def pack_artifact(
    artifact: Artifact,
    *,
    transport: str = "inline_bytes",
    max_bytes: Optional[int] = None,
) -> Artifact:
    """Return ``artifact`` with ``data_b64`` populated according to ``transport``.

    ``transport="path"`` returns the artifact unchanged (no inlining).
    ``transport="inline_bytes"`` reads the file, checks the size budget, and
    returns a new Artifact with ``data_b64`` set. Raises
    :class:`ArtifactTooLargeError` if the file exceeds the budget.
    """
    if transport == "path":
        return Artifact(
            kind=artifact.kind,
            path=artifact.path,
            mime=artifact.mime,
            width=artifact.width,
            height=artifact.height,
            num_frames=artifact.num_frames,
            transport="path",
            data_b64=None,
        )
    if transport != "inline_bytes":
        raise ValueError(f"Unknown artifact transport: {transport!r}")

    source = Path(artifact.path)
    if not source.exists():
        raise FileNotFoundError(f"Artifact source missing: {source}")
    size = source.stat().st_size
    limit = max_bytes if max_bytes is not None else inline_limit_bytes()
    if size > limit:
        raise ArtifactTooLargeError(path=str(source), size_bytes=size, max_bytes=limit)

    data = source.read_bytes()
    return Artifact(
        kind=artifact.kind,
        path=str(source.name),  # keep just the filename for the client's sake
        mime=artifact.mime,
        width=artifact.width,
        height=artifact.height,
        num_frames=artifact.num_frames,
        transport="inline_bytes",
        data_b64=base64.b64encode(data).decode("ascii"),
    )


def unpack_artifact(artifact: Artifact, *, output_dir: Path) -> Artifact:
    """Client-side: materialize an inline-bytes artifact to disk.

    Returns a new Artifact with ``transport="path"`` pointing at the
    freshly-written file. If the artifact is already ``transport="path"``,
    returns it unchanged.
    """
    if artifact.transport == "path":
        return artifact
    if artifact.transport != "inline_bytes":
        raise ValueError(f"Unknown artifact transport: {artifact.transport!r}")
    if not artifact.data_b64:
        raise ValueError(f"Inline-bytes artifact {artifact.path!r} has no payload")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / Path(artifact.path).name
    target.write_bytes(base64.b64decode(artifact.data_b64))
    return Artifact(
        kind=artifact.kind,
        path=str(target),
        mime=artifact.mime,
        width=artifact.width,
        height=artifact.height,
        num_frames=artifact.num_frames,
        transport="path",
        data_b64=None,
    )
