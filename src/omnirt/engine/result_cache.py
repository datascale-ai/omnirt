"""Cross-request cache helpers for reusable intermediate results."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
import threading
from typing import Any, Dict, Optional

from omnirt.core.types import GenerateRequest


class ResultCache:
    def __init__(self, *, max_items: int = 256) -> None:
        self.max_items = max(int(max_items), 1)
        self._items: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def lookup_embeddings(self, request: GenerateRequest) -> Optional[Dict[str, Any]]:
        key = self._embedding_key(request)
        with self._lock:
            bundle = self._items.get(key)
            if bundle is None:
                return None
            self._items.move_to_end(key)
            return dict(bundle)

    def save_embeddings(self, request: GenerateRequest, bundle: Dict[str, Any]) -> None:
        key = self._embedding_key(request)
        with self._lock:
            self._items[key] = dict(bundle)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def _embedding_key(self, request: GenerateRequest) -> str:
        payload = {
            "task": request.task,
            "model": request.model,
            "prompt": request.inputs.get("prompt"),
            "negative_prompt": request.inputs.get("negative_prompt"),
            "dtype": request.config.get("dtype"),
            "model_path": request.config.get("model_path"),
            "max_sequence_length": request.config.get("max_sequence_length"),
        }
        return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
