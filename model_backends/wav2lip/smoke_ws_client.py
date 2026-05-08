#!/usr/bin/env python3
"""Minimal client: init + one silent AUDI chunk (requires same deps as server + pillow not needed)."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC_AUDIO = b"AUDI"


async def main(url: str, ref_image: Path) -> int:
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        print("pip install websockets", file=sys.stderr)
        return 2

    img_bytes = ref_image.read_bytes()
    async with connect(url, max_size=50 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "init",
                    "ref_image": base64.b64encode(img_bytes).decode(),
                    "prompt": "test",
                    "seed": 42,
                }
            )
        )
        resp = json.loads(await ws.recv())
        if resp.get("type") == "error":
            print("init error:", resp, file=sys.stderr)
            return 1
        print("init_ok:", resp)
        sl = int(resp["slice_len"])
        fps = int(resp["fps"])
        chunk_samples = sl * 16000 // fps
        pcm = np.zeros(chunk_samples, dtype=np.int16)
        await ws.send(MAGIC_AUDIO + pcm.tobytes())
        vid = await ws.recv()
        if isinstance(vid, str):
            print("generate error:", vid, file=sys.stderr)
            return 1
        if len(vid) < 8 or vid[:4] != b"VIDX":
            print("unexpected reply", vid[:32], file=sys.stderr)
            return 1
        n = struct.unpack("<I", vid[4:8])[0]
        print("VIDX frames:", n, "total_bytes:", len(vid))
        await ws.send(json.dumps({"type": "close"}))
        clo = json.loads(await ws.recv())
        print("close:", clo)
    return 0


def _default_ref_image() -> Path:
    omnirt_root = Path(__file__).resolve().parents[2]
    candidates = [
        omnirt_root / "models" / "repos" / "Wav2Lip" / "examples" / "man.png",
        omnirt_root.parent / "opentalking" / "examples" / "avatars" / "demo-musetalk" / "preview.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument(
        "--ref-image",
        type=Path,
        default=_default_ref_image(),
        help="Reference face image (PNG/JPEG)",
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.url, args.ref_image)))
