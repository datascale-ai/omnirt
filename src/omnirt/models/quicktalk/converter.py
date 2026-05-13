"""Convert QuickTalk's recovered ONNX network into a TorchScript checkpoint."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def default_quicktalk_model_root() -> Path:
    model_root = os.environ.get("OMNIRT_MODEL_ROOT", "").strip()
    if model_root:
        return Path(model_root).expanduser().resolve() / "quicktalk"
    return (Path.cwd() / "models" / "quicktalk").resolve()


@dataclass(frozen=True)
class ConversionParity:
    output_name: str
    max_abs_error: float
    mean_abs_error: float


@dataclass(frozen=True)
class QuickTalkRuntimePaths:
    checkpoints: Path
    repair_path: Path
    hubert_path: Path
    aux_root: Path


class QuickTalkOnnxLSTMModule:
    """Placeholder name for type checkers; real base class is torch.nn.Module."""


def _onnx_lstm_module_class():
    import torch
    from torch import nn

    class OnnxLSTM(nn.Module):
        def __init__(
            self,
            *,
            weight_ih: torch.Tensor,
            weight_hh: torch.Tensor,
            bias_ih: torch.Tensor,
            bias_hh: torch.Tensor,
            hidden_size: int,
        ) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.register_buffer("weight_ih", weight_ih.contiguous())
            self.register_buffer("weight_hh", weight_hh.contiguous())
            self.register_buffer("bias_ih", bias_ih.contiguous())
            self.register_buffer("bias_hh", bias_hh.contiguous())

        @staticmethod
        def onnx_to_torch_gate_order(value: torch.Tensor) -> torch.Tensor:
            i, o, f, c = torch.chunk(value, 4, dim=0)
            return torch.cat((i, f, c, o), dim=0)

        @classmethod
        def from_onnx_tensors(
            cls,
            *,
            weight: torch.Tensor,
            recurrence: torch.Tensor,
            bias: torch.Tensor | None,
            hidden_size: int,
        ) -> "OnnxLSTM":
            if weight.shape[0] != 1 or recurrence.shape[0] != 1:
                raise NotImplementedError("QuickTalk converter supports one-direction ONNX LSTM only.")
            w = cls.onnx_to_torch_gate_order(weight[0].float())
            r = cls.onnx_to_torch_gate_order(recurrence[0].float())
            if bias is None:
                b_w = torch.zeros(4 * hidden_size, dtype=torch.float32)
                b_r = torch.zeros(4 * hidden_size, dtype=torch.float32)
            else:
                if bias.shape[0] != 1:
                    raise NotImplementedError("QuickTalk converter supports one-direction ONNX LSTM bias only.")
                raw_bias = bias[0].float()
                b_w = cls.onnx_to_torch_gate_order(raw_bias[: 4 * hidden_size])
                b_r = cls.onnx_to_torch_gate_order(raw_bias[4 * hidden_size : 8 * hidden_size])
            return cls(
                weight_ih=w,
                weight_hh=r,
                bias_ih=b_w,
                bias_hh=b_r,
                hidden_size=hidden_size,
            )

        def forward(
            self,
            x: torch.Tensor,
            initial_h: torch.Tensor | None = None,
            initial_c: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if x.dim() != 3:
                raise RuntimeError(f"QuickTalk LSTM expected 3D input, got {tuple(x.shape)}")
            seq_len, batch, _ = x.shape
            if initial_h is None:
                h_t = x.new_zeros((batch, self.hidden_size))
            else:
                h_t = initial_h[0].to(dtype=x.dtype)
            if initial_c is None:
                c_t = x.new_zeros((batch, self.hidden_size))
            else:
                c_t = initial_c[0].to(dtype=x.dtype)
            weight_ih = self.weight_ih.to(dtype=x.dtype)
            weight_hh = self.weight_hh.to(dtype=x.dtype)
            bias_ih = self.bias_ih.to(dtype=x.dtype)
            bias_hh = self.bias_hh.to(dtype=x.dtype)
            outputs: list[torch.Tensor] = []
            for t in range(seq_len):
                gates = (
                    x[t].matmul(weight_ih.t())
                    + bias_ih
                    + h_t.matmul(weight_hh.t())
                    + bias_hh
                )
                i, f, g, o = gates.chunk(4, dim=1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                c_t = f * c_t + i * g
                h_t = o * torch.tanh(c_t)
                outputs.append(h_t)
            y = torch.stack(outputs, dim=0).unsqueeze(1)
            return y, h_t.unsqueeze(0), c_t.unsqueeze(0)

    OnnxLSTM.__name__ = "QuickTalkOnnxLSTM"
    OnnxLSTM.__qualname__ = "QuickTalkOnnxLSTM"
    OnnxLSTM.__module__ = __name__
    globals()["QuickTalkOnnxLSTM"] = OnnxLSTM
    return OnnxLSTM


QuickTalkOnnxLSTM: Any | None = None


def _ensure_lstm_module_class():
    global QuickTalkOnnxLSTM
    if QuickTalkOnnxLSTM is None:
        QuickTalkOnnxLSTM = _onnx_lstm_module_class()
    return QuickTalkOnnxLSTM


def ensure_quicktalk_pickle_types():
    """Register converter-defined module classes before loading torch pickles."""

    return _ensure_lstm_module_class()


def quicktalk_checkpoint_path(
    model_root: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Path:
    if checkpoint is not None and str(checkpoint).strip():
        return Path(checkpoint).expanduser().resolve()
    model_root = model_root or default_quicktalk_model_root()
    return Path(model_root).expanduser().resolve() / "quicktalk.pth"


def _default_onnx_path(model_root: str | Path) -> Path:
    root = Path(model_root).expanduser().resolve()
    for candidate in (
        root / "checkpoints" / "256.onnx",
        root / "256.onnx",
    ):
        if candidate.is_file():
            return candidate
    return root / "checkpoints" / "256.onnx"


def resolve_quicktalk_runtime_paths(model_root: str | Path) -> QuickTalkRuntimePaths:
    root = Path(model_root).expanduser().resolve()
    checkpoints = root / "checkpoints"

    def prefer_checkpoint(name: str) -> Path:
        checkpoint_path = checkpoints / name
        root_path = root / name
        if checkpoint_path.exists() or not root_path.exists():
            return checkpoint_path
        return root_path

    repair_path = prefer_checkpoint("repair.npy")
    hubert_path = prefer_checkpoint("chinese-hubert-large")
    aux_min = checkpoints / "auxiliary_min"
    if aux_min.exists():
        aux_root = aux_min
    else:
        aux_root = prefer_checkpoint("auxiliary")
    return QuickTalkRuntimePaths(
        checkpoints=checkpoints,
        repair_path=repair_path,
        hubert_path=hubert_path,
        aux_root=aux_root,
    )


def _random_inputs(input_shapes: Sequence[Sequence[int]]) -> list[np.ndarray]:
    rng = np.random.default_rng(1234)
    arrays: list[np.ndarray] = []
    for shape in input_shapes:
        normalized: list[int] = []
        for dim in shape:
            try:
                value = int(dim)
            except (TypeError, ValueError):
                value = 1
            normalized.append(1 if value <= 0 else value)
        arrays.append(rng.standard_normal(normalized).astype(np.float32))
    return arrays


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _register_lstm_converter() -> None:
    """Register the small ONNX LSTM subset used by QuickTalk with onnx2torch."""

    from onnx2torch.node_converters.registry import add_converter, get_converter
    from onnx2torch.onnx_graph import OnnxGraph
    from onnx2torch.onnx_node import OnnxNode
    from onnx2torch.utils.common import OnnxMapping, OperationConverterResult

    try:
        get_converter("LSTM", 7)
        return
    except NotImplementedError:
        pass

    lstm_cls = _ensure_lstm_module_class()

    @add_converter(operation_type="LSTM", version=7)
    @add_converter(operation_type="LSTM", version=14)
    def _convert_lstm(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
        hidden_size = int(node.attributes["hidden_size"])
        direction = str(node.attributes.get("direction", "forward"))
        if direction != "forward":
            raise NotImplementedError("QuickTalk converter supports forward ONNX LSTM only.")
        inputs = list(node.input_values)
        weight_name = inputs[1]
        recurrence_name = inputs[2]
        bias_name = inputs[3] if len(inputs) > 3 else ""
        if weight_name not in graph.initializers or recurrence_name not in graph.initializers:
            raise NotImplementedError("QuickTalk converter requires constant LSTM weights.")
        weight = graph.initializers[weight_name].to_torch()
        recurrence = graph.initializers[recurrence_name].to_torch()
        bias = graph.initializers[bias_name].to_torch() if bias_name in graph.initializers else None
        module = lstm_cls.from_onnx_tensors(
            weight=weight,
            recurrence=recurrence,
            bias=bias,
            hidden_size=hidden_size,
        )
        mapped_inputs: list[str] = [inputs[0]]
        if len(inputs) > 5 and inputs[5]:
            mapped_inputs.append(inputs[5])
        if len(inputs) > 6 and inputs[6]:
            mapped_inputs.append(inputs[6])
        return OperationConverterResult(
            torch_module=module,
            onnx_mapping=OnnxMapping(inputs=tuple(mapped_inputs), outputs=node.output_values),
        )


def _default_trace_inputs() -> tuple[Any, ...]:
    import torch

    return (
        torch.zeros(1, 10, 1024, dtype=torch.float32),
        torch.zeros(1, 7, 256, 256, dtype=torch.float32),
        torch.zeros(2, 1, 576, dtype=torch.float32),
        torch.zeros(2, 1, 576, dtype=torch.float32),
    )


def convert_quicktalk_onnx_to_pth(
    *,
    onnx_path: str | Path,
    output_path: str | Path,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    verify: bool = True,
) -> list[ConversionParity]:
    """Convert ONNX to a TorchScript checkpoint and optionally verify parity.

    Conversion uses ``onnx2torch`` offline, then traces the resulting module and
    saves a TorchScript archive. Serving loads the archive with ``torch.jit.load``
    and does not need ONNX, ONNX Runtime, or onnx2torch installed.
    """

    import onnx
    import onnxruntime as ort
    import torch
    from onnx2torch import convert

    onnx_path = Path(onnx_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"QuickTalk ONNX checkpoint not found: {onnx_path}")

    _register_lstm_converter()
    onnx_model = onnx.load(str(onnx_path))
    torch_model = convert(onnx_model).eval()

    parity: list[ConversionParity] = []
    if verify:
        ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = ort_session.get_inputs()
        inputs = _random_inputs([item.shape for item in ort_inputs])
        feed = {item.name: array for item, array in zip(ort_inputs, inputs)}
        expected = ort_session.run(None, feed)
        with torch.inference_mode():
            actual_raw = torch_model(*[torch.from_numpy(array) for array in inputs])
        actual_values = actual_raw if isinstance(actual_raw, (tuple, list)) else (actual_raw,)
        if len(actual_values) != len(expected):
            raise RuntimeError(
                f"Converted QuickTalk output count mismatch: torch={len(actual_values)} onnx={len(expected)}"
            )
        output_names = [item.name for item in ort_session.get_outputs()]
        for name, actual, want in zip(output_names, actual_values, expected):
            actual_np = _as_numpy(actual).astype(np.float32, copy=False)
            want_np = np.asarray(want, dtype=np.float32)
            if actual_np.shape != want_np.shape:
                raise RuntimeError(
                    f"Converted QuickTalk output {name} shape mismatch: "
                    f"torch={actual_np.shape} onnx={want_np.shape}"
                )
            abs_err = np.abs(actual_np - want_np)
            max_abs = float(abs_err.max(initial=0.0))
            mean_abs = float(abs_err.mean()) if abs_err.size else 0.0
            if not np.allclose(actual_np, want_np, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"Converted QuickTalk output {name} failed parity: "
                    f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} atol={atol} rtol={rtol}"
                )
            parity.append(
                ConversionParity(
                    output_name=name,
                    max_abs_error=max_abs,
                    mean_abs_error=mean_abs,
                )
            )

    trace_inputs = _default_trace_inputs()
    with torch.inference_mode():
        eager_trace_outputs = torch_model(*trace_inputs)
    traced_model = torch.jit.trace(torch_model, trace_inputs, strict=False)
    with torch.inference_mode():
        traced_outputs = traced_model(*trace_inputs)
    eager_values = eager_trace_outputs if isinstance(eager_trace_outputs, (tuple, list)) else (eager_trace_outputs,)
    traced_values = traced_outputs if isinstance(traced_outputs, (tuple, list)) else (traced_outputs,)
    if len(eager_values) != len(traced_values):
        raise RuntimeError(
            f"Traced QuickTalk output count mismatch: eager={len(eager_values)} traced={len(traced_values)}"
        )
    for idx, (eager, traced) in enumerate(zip(eager_values, traced_values)):
        eager_np = _as_numpy(eager).astype(np.float32, copy=False)
        traced_np = _as_numpy(traced).astype(np.float32, copy=False)
        if eager_np.shape != traced_np.shape:
            raise RuntimeError(
                f"Traced QuickTalk output {idx} shape mismatch: "
                f"eager={eager_np.shape} traced={traced_np.shape}"
            )
        if not np.allclose(eager_np, traced_np, atol=atol, rtol=rtol):
            abs_err = np.abs(eager_np - traced_np)
            raise RuntimeError(
                f"Traced QuickTalk output {idx} failed parity: "
                f"max_abs={float(abs_err.max(initial=0.0)):.6g} "
                f"mean_abs={float(abs_err.mean()) if abs_err.size else 0.0:.6g} "
                f"atol={atol} rtol={rtol}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    return parity


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default=str(default_quicktalk_model_root()))
    parser.add_argument("--onnx", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    onnx_path = Path(args.onnx).expanduser().resolve() if args.onnx else _default_onnx_path(args.model_root)
    output_path = quicktalk_checkpoint_path(args.model_root, args.output or None)
    parity = convert_quicktalk_onnx_to_pth(
        onnx_path=onnx_path,
        output_path=output_path,
        atol=args.atol,
        rtol=args.rtol,
        verify=not args.no_verify,
    )
    for item in parity:
        print(
            f"{item.output_name}: max_abs={item.max_abs_error:.6g} "
            f"mean_abs={item.mean_abs_error:.6g}",
            flush=True,
        )
    print(f"saved={output_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
