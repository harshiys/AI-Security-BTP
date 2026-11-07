"""Static weight-space features for LoRA adapters (no base model, no execution).

For each (layer, module) pair with factors A (r x in) and B (out x r), the update is dW = scale * B @ A.
Its singular values are obtained cheaply from the r x r matrix Rb @ Ra.T, where
B = Qb Rb and A.T = Qa Ra  (so dW = Qb (Rb Ra.T) Qa.T and Qb, Qa have orthonormal columns).
AdaLoRA's diagonal E is folded into A (rows scaled by E).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .safeload import load_lora_tensors

METRICS = [
    "log_sigma1", "log_fro", "top1_energy", "top10pct_energy", "top25pct_energy",
    "spec_entropy", "eff_rank", "stable_rank", "tail_ratio", "log_cond",
    "kurt_A", "kurt_B", "log_std_A", "log_std_B", "absmax_B",
]
GROUPS = ("q", "v")
STATS = ("mean", "std", "min", "max", "early", "mid", "late")

_KEY = re.compile(r"^(?P<prefix>.+)\.lora_(?P<kind>[ABE])(?:\.[A-Za-z0-9_]+)?(?:\.weight)?$")
_LAYER = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks?)\.(\d+)(?:\.|$)")


def feature_names() -> list[str]:
    return [f"{g}__{m}__{s}" for g in GROUPS for m in METRICS for s in STATS]


def module_group(prefix: str) -> str:
    tail = prefix.rsplit(".", 1)[-1].lower()
    if tail in ("q_proj", "q", "query", "wq", "query_proj"):
        return "q"
    if tail in ("v_proj", "v", "value", "wv", "value_proj"):
        return "v"
    return "other"


def collect_pairs(tensors: dict[str, np.ndarray]):
    by: dict[str, dict[str, np.ndarray]] = {}
    for k, v in tensors.items():
        m = _KEY.match(k)
        if m:
            by.setdefault(m["prefix"], {})[m["kind"]] = v
    return [(p, d["A"], d["B"], d.get("E")) for p, d in by.items() if "A" in d and "B" in d]


def read_adapter_scale(cfg_path: str | Path | None) -> tuple[float, str]:
    """alpha/r from adapter_config.json if present, else 1.0 (and say so)."""
    if not cfg_path:
        return 1.0, "none"
    try:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        r, alpha = cfg.get("r"), cfg.get("lora_alpha")
        if r and alpha:
            return (alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r), "adapter_config"
    except Exception:  # noqa: BLE001
        pass
    return 1.0, "none"


def _kurt(x: np.ndarray) -> float:
    x = x.ravel().astype(np.float64)
    d = x - x.mean()
    v = float((d * d).mean())
    return 0.0 if v <= 0 else float((d ** 4).mean() / (v * v) - 3.0)


def _log10(x: float) -> float:
    return math.log10(max(float(x), 1e-12))


def pair_features(A, B, E=None, scale: float = 1.0) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
        raise ValueError(f"incompatible LoRA shapes A{A.shape} B{B.shape}")
    A_eff = A
    if E is not None:
        E = np.asarray(E, dtype=np.float32).reshape(-1)
        if E.shape[0] != A.shape[0]:
            raise ValueError(f"E length {E.shape[0]} != rank {A.shape[0]}")
        A_eff = A * E[:, None]
    r = A.shape[0]
    Rb = np.linalg.qr(B, mode="r")
    Ra = np.linalg.qr(A_eff.T, mode="r")
    s = np.linalg.svd((Rb @ Ra.T).astype(np.float64), compute_uv=False) * scale
    e = s ** 2
    tot = float(e.sum())
    f = {m: 0.0 for m in METRICS}
    f["kurt_A"], f["kurt_B"] = _kurt(A), _kurt(B)
    f["log_std_A"], f["log_std_B"] = _log10(A.std()), _log10(B.std())
    bs = float(B.std())
    f["absmax_B"] = float(np.abs(B).max() / bs) if bs > 0 else 0.0
    if tot <= 0:
        return f  # degenerate (e.g. B == 0): spectral features stay 0
    p = e / tot
    k10, k25 = max(1, math.ceil(0.1 * r)), max(1, math.ceil(0.25 * r))  # rank-relative, never saturate
    pp = p[p > 0]
    H = float(-(pp * np.log(pp)).sum())
    f.update(
        log_sigma1=_log10(s[0]), log_fro=_log10(math.sqrt(tot)),
        top1_energy=float(p[0]), top10pct_energy=float(p[:k10].sum()),
        top25pct_energy=float(p[:k25].sum()),
        spec_entropy=H / math.log(r) if r > 1 else 0.0,
        eff_rank=math.exp(H) / r, stable_rank=(tot / float(e[0])) / r,
        tail_ratio=float(s[min(r // 2, len(s) - 1)] / s[0]),
        log_cond=min(12.0, max(0.0, math.log10((s[0] + 1e-30) / (s[-1] + 1e-30)))),
    )
    return f


def aggregate(rows: list[dict]) -> dict[str, float]:
    """Fixed-length vector regardless of rank / depth: mean,std,min,max + early/mid/late thirds."""
    feats: dict[str, float] = {}
    for g in GROUPS:
        rr = sorted((x for x in rows if x["module"] == g), key=lambda x: (x["layer"], x["prefix"]))
        for m in METRICS:
            vals = np.array([x[m] for x in rr], dtype=np.float64)
            if len(vals) == 0:
                stats = [0.0] * len(STATS)
            else:
                thirds = [c.mean() if len(c) else vals.mean() for c in np.array_split(vals, 3)]
                stats = [vals.mean(), vals.std(), vals.min(), vals.max(), *thirds]
            for s, v in zip(STATS, stats):
                feats[f"{g}__{m}__{s}"] = float(v)
    return feats


def extract_adapter(weights_path: str | Path, cfg_path: str | Path | None = None):
    """Returns (meta, adapter_features, per_layer_rows). Raises on anything unsafe/unsupported."""
    tensors = load_lora_tensors(weights_path)
    pairs = collect_pairs(tensors)
    if not pairs:
        sample = list(tensors)[:5] or "no tensors with 'lora_' in their name"
        raise ValueError(f"no lora_A/lora_B pairs found (sample keys: {sample})")
    scale, scale_src = read_adapter_scale(cfg_path)
    rows = []
    for prefix, A, B, E in pairs:
        lm = _LAYER.search(prefix)
        row = {"prefix": prefix, "module": module_group(prefix),
               "layer": int(lm.group(1)) if lm else -1, "rank": int(A.shape[0])}
        row.update(pair_features(A, B, E, scale))
        rows.append(row)
    qv = [r for r in rows if r["module"] in GROUPS]
    if not qv:
        raise ValueError(f"no q/v LoRA pairs among {len(rows)} pairs (prefix sample: {rows[0]['prefix']})")
    meta = {
        "n_pairs_q": sum(r["module"] == "q" for r in rows),
        "n_pairs_v": sum(r["module"] == "v" for r in rows),
        "n_pairs_other": sum(r["module"] == "other" for r in rows),
        "rank_detected": int(np.median([r["rank"] for r in qv])),
        "n_layers": len({r["layer"] for r in qv}),
        "scale": float(scale), "scale_source": scale_src,
    }
    return meta, aggregate(rows), qv
