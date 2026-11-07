#!/usr/bin/env python3
"""Stage 2U - UNLABELED local feature extraction (no manifest, no labels, no network, no base model).

Scans a folder on disk for `adapter_model.safetensors` files (e.g. ones the earlier stages already downloaded into
work/hf_files/...), reads the `adapter_config.json` next to each one, and runs the SAME extractor as Stage 2
(c1.features.extract_adapter -> c1.safeload safe reader). One row per adapter goes into a SEPARATE CSV:

    work/features_unlabeled/features_unlabeled.csv

This mode assigns NO labels. It never touches work/manifest.csv or work/features/ (the verified-label pipeline), and
s3/s4 do not read its output. Use it to check that real adapters parse and to look at their features while the
meaning of PADBench label0/label1 is still unverified.

    python s2u_extract_unlabeled.py --limit 2        # first: 2 adapters
    python s2u_extract_unlabeled.py                  # then: everything under the folder (re-runs resume)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

from c1.features import extract_adapter, feature_names

WEIGHT_NAME = "adapter_model.safetensors"
DEFAULT_DIR = "work/hf_files/llama2_7b_toxic_backdoors_hard_rank64_qv"
DEFAULT_OUT = "work/features_unlabeled/features_unlabeled.csv"
BASE_COLS = ["item_path", "local_path", "status", "error", "seconds", "size_bytes", "config_found",
             "n_pairs_q", "n_pairs_v", "n_pairs_other", "rank_detected", "n_layers", "scale", "scale_source"]
COLS = BASE_COLS + feature_names()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="folder to scan recursively (default: %(default)s)")
    ap.add_argument("--base", default="work/hf_files",
                    help="item_path is recorded relative to this folder, so it equals the repo path "
                         "(default: %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N not-yet-extracted adapters (0 = all)")
    args = ap.parse_args()

    scan, base, out = Path(args.dir), Path(args.base), Path(args.out)
    if out.as_posix().endswith("features/features.csv"):
        sys.exit("refusing to write into the labeled pipeline's cache (work/features/features.csv); use another --out")
    if not scan.is_dir():
        sys.exit(f"scan folder not found: {scan}  (run from the project root, or pass --dir)")

    found = sorted(scan.rglob(WEIGHT_NAME))
    print("UNLABELED MODE: no labels are assigned; output is not used by s3/s4.")
    if not found:
        names = sorted({p.name for p in scan.rglob("*") if p.is_file()})[:10]
        sys.exit(f"no '{WEIGHT_NAME}' under {scan}. File names seen there: {names or 'none'}")

    def item_path(p: Path) -> str:
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            return p.relative_to(scan.parent).as_posix()

    done: set[str] = set()
    if out.exists():
        old = pd.read_csv(out)
        if list(old.columns) != COLS:
            sys.exit(f"{out} has different columns than this script writes; move it aside or use --out")
        done = set(old.loc[old["status"] == "ok", "item_path"])
    todo = [p for p in found if item_path(p) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(found)} adapters found under {scan} | {len(done)} already extracted | {len(todo)} to do")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for i, p in enumerate(todo, 1):
        cfg = p.with_name("adapter_config.json")
        rec = {c: math.nan for c in COLS}
        rec.update(item_path=item_path(p), local_path=str(p), error="", size_bytes=p.stat().st_size,
                   config_found=cfg.exists())
        t0 = time.time()
        try:
            meta, feats, _layer_rows = extract_adapter(p, cfg if cfg.exists() else None)
            rec.update(meta)
            rec.update(feats)
            rec["status"] = "ok"
            n_ok += 1
        except Exception as e:  # noqa: BLE001 - record the failure for this adapter and carry on
            rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"[:300]
        rec["seconds"] = round(time.time() - t0, 2)
        pd.DataFrame([rec], columns=COLS).to_csv(out, mode="a", header=not out.exists(), index=False)
        if rec["status"] == "ok":
            print(f"[{i}/{len(todo)}] ok    {rec['seconds']:5.1f}s rank={rec['rank_detected']} "
                  f"pairs(q,v)=({rec['n_pairs_q']},{rec['n_pairs_v']}) layers={rec['n_layers']} "
                  f"scale={rec['scale']:.3g}({rec['scale_source']})  {rec['item_path']}")
        else:
            print(f"[{i}/{len(todo)}] ERROR {rec['seconds']:5.1f}s {rec['item_path']}\n          {rec['error']}")

    res = pd.read_csv(out).drop_duplicates("item_path", keep="last")
    print(f"\nstatus counts in {out}: {res['status'].value_counts().to_dict()}")
    if todo and n_ok == 0:
        print("every attempted adapter failed; see the error lines above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
