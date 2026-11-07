"""Safe, model-free loading of LoRA adapter tensors.

* .safetensors: parsed with a small pure-numpy reader (no pickle, no torch, no code execution).
  Only tensors whose name contains 'lora_' are read from disk.
* .bin/.pt/.pth: torch.load(weights_only=True). If a file contains anything other than plain
  tensors/containers (i.e. a real pickle payload), loading is refused and the adapter is recorded
  as an error - we never fall back to an unrestricted pickle load.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_FLOATS = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
_ITEMSIZE = {"F32": 4, "F16": 2, "F64": 8, "BF16": 2}


class UnsafeOrUnsupportedFile(Exception):
    pass


def _want(name: str) -> bool:
    return "lora_" in name


def read_safetensors(path: str | Path, keep=_want) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise UnsafeOrUnsupportedFile("file too short for a safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 200_000_000:
            raise UnsafeOrUnsupportedFile(f"implausible safetensors header length {n}")
        try:
            header = json.loads(f.read(n))
        except Exception as e:  # noqa: BLE001
            raise UnsafeOrUnsupportedFile(f"unparseable safetensors header: {e}") from e
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__" or not keep(name):
                continue
            dt, shape = info["dtype"], info["shape"]
            if dt not in _ITEMSIZE:
                continue  # skip non-float tensors
            s, e = info["data_offsets"]
            count = int(np.prod(shape)) if shape else 1
            if e - s != count * _ITEMSIZE[dt]:
                raise UnsafeOrUnsupportedFile(f"offset/shape mismatch for {name}")
            f.seek(base + s)
            buf = f.read(e - s)
            if dt == "BF16":
                arr = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                arr = np.frombuffer(buf, dtype=_FLOATS[dt])
            out[name] = arr.reshape(shape).astype(np.float32, copy=False)
    return out


def read_torch_weights_only(path: str | Path, keep=_want) -> dict[str, np.ndarray]:
    try:
        import torch
    except ImportError as e:
        raise UnsafeOrUnsupportedFile("torch is required for .bin/.pt adapters (pip install torch)") from e
    try:
        try:
            obj = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError:  # legacy (non-zip) files cannot be mmapped
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:  # pickle.UnpicklingError etc.
        raise UnsafeOrUnsupportedFile(f"refused: not a plain tensor file under weights_only=True ({e})") from e
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise UnsafeOrUnsupportedFile(f"unexpected top-level object {type(obj).__name__}")
    out = {}
    for k, v in obj.items():
        if isinstance(k, str) and keep(k) and hasattr(v, "float"):
            out[k] = v.detach().float().cpu().numpy()
    return out


def load_lora_tensors(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if p.suffix.lower() == ".safetensors":
        return read_safetensors(p)
    return read_torch_weights_only(p)


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray], bf16: bool = False) -> None:
    """Minimal writer (used only by the self-test to create synthetic adapters)."""
    header, chunks, off = {}, [], 0
    for k, a in tensors.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        if bf16:
            b = (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()
            dt = "BF16"
        else:
            b, dt = a.tobytes(), "F32"
        header[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        chunks.append(b)
        off += len(b)
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for c in chunks:
            f.write(c)
