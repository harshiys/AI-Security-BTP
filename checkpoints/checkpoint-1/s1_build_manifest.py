#!/usr/bin/env python3
"""Stage 1 - find the PADBench adapters that actually exist, label them, verify, sample, write the manifest.

Sources:
  --source hf     list the Hugging Face repo (no weights downloaded here)
  --source local  use files already on disk. Default root = <work>/hf_files (where stage 2 stores downloads);
                  override with --local-root. Config folders are searched at ANY depth below that root.
If --folders is omitted: hf = the default rank folders; local = every config folder that really contains weights.

Labels come from label_config.json (project convention, mapping_status=provisional) or <work>/label_rules.json if
you made one. Output: <work>/manifest.csv, <work>/manifest_report.json
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

from c1.common import (DEFAULT_FOLDERS, TAG_LOCAL_REAL, TAG_REAL, TAG_SYNTHETIC, FolderResolutionError,
                       describe_tree, discover_config_folders, is_real_tag, is_weight_file, make_source, norm_name,
                       resolve_folders, weight_name_matches)
from c1.labels import LabelError, LabelResolver, load_rules, relpath_in_folder, verify_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["hf", "local"], default="hf")
ap.add_argument("--local-root", help="folder containing the PADBench config folders (default: <work>/hf_files)")
ap.add_argument("--work", default="work")
ap.add_argument("--rules", default=None, help="default: <work>/label_rules.json if it exists, else label_config.json")
ap.add_argument("--revision", default=None, help="pin a dataset commit sha (default: current; recorded in the report)")
ap.add_argument("--weight-name", default="*.safetensors",
                help="file-name pattern for adapter weights (default *.safetensors; e.g. adapter_model.safetensors)")
ap.add_argument("--data-tag", default=None, help="provenance tag (default: padbench_hf / padbench_local / "
                                                 "synthetic_selftest if the folder carries the synthetic marker)")
ap.add_argument("--folders", nargs="*", default=None, help="config folder names (default: see above)")
ap.add_argument("--root-prefix", default=None, help="only accept config folders whose parent is this path")
ap.add_argument("--max-per-class-per-folder", type=int, default=30)
ap.add_argument("--min-per-class", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--drop-unlabeled", action="store_true", help="drop adapters no rule matches (default: error)")
ap.add_argument("--allow-contradictions", action="store_true")
args = ap.parse_args()

work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
is_local = args.source == "local"
names_arg = [norm_name(x) for x in args.folders] if args.folders else None
src = make_source(args.source, args.local_root, work, args.revision, names_arg)
revision = getattr(src, "revision", None)
fixture = is_local and src.is_synthetic()
data_tag = args.data_tag or (TAG_SYNTHETIC if fixture else (TAG_LOCAL_REAL if is_local else TAG_REAL))
if args.source == "hf" and data_tag != TAG_REAL:
    sys.exit(f"--source hf must use data tag '{TAG_REAL}' (got '{data_tag}')")
if is_local and data_tag == TAG_REAL:
    sys.exit(f"tag '{TAG_REAL}' is reserved for the Hugging Face source; use '{TAG_LOCAL_REAL}' for local files")
if fixture and is_real_tag(data_tag):
    sys.exit(f"this folder carries the synthetic-fixture marker, so it cannot be tagged '{data_tag}' (real PADBench)")
if is_local and not fixture and data_tag == TAG_SYNTHETIC:
    sys.exit("the synthetic tag is only for the fake fixtures under tests/; this folder has no synthetic marker")


def folder_hints(name: str) -> dict:
    n = name.lower()
    rk = re.search(r"rank(\d+)", n)
    attack = next((a for a in ("ripple", "insertsent", "stybkd", "syntactic", "toxic_backdoors_hard",
                               "toxic_backdoors_alpaca", "toxic_backdoors") if a in n), "")
    method = next((m for m in ("adalora", "dora", "lora_plus", "qlora") if m in n), "lora")
    # family = everything except the rank: same base model / dataset / attack / PEFT method. Different ranks of one
    # family are NOT independent (same recipe), so s3 can group by family to test genuine generalisation.
    family = re.sub(r"_rank\d+", "", name)
    return {"rank_hint": int(rk.group(1)) if rk else -1, "attack_hint": attack, "method_hint": method,
            "family": family}


try:
    default_rules = work / "label_rules.json"
    if not default_rules.exists():
        default_rules = Path(__file__).resolve().parent / "label_config.json"
    rules_path = args.rules or str(default_rules)
    spec = load_rules(rules_path)
    resolver = LabelResolver(spec, src)
    rng = np.random.default_rng(args.seed)
    rows, problems, incomplete = [], {}, []
    multi_weight_dirs = 0

    # ---- 1. find the config folders that really exist ----
    try:
        if names_arg is None and is_local:
            resolved = discover_config_folders(src, args.weight_name)
            names = sorted(resolved)
            prefix = ""
            missing = [] if names else ["(none found)"]
        else:
            names = names_arg or list(DEFAULT_FOLDERS)
            prefix, resolved, missing = resolve_folders(src, names, args.root_prefix)
    except FolderResolutionError as e:
        raise LabelError(str(e))
    if missing:
        where = describe_tree(src) if is_local else f"  top-level folders: {list(src.iter_folders(''))[:30]}"
        print("DATASET DISCOVERY FAILED: could not find "
              f"{missing if names_arg or not is_local else 'any PADBench adapter folders'} under "
              f"{getattr(src, 'root', 'the Hugging Face repo')}.\nWhat is actually there:\n{where}\n"
              "Fix: point --local-root at the folder that contains the config folders (or any parent of it), or pass "
              "--folders with the real folder names. Nothing was guessed.", file=sys.stderr)
        sys.exit(2)
    print("config folders used (name -> path under the source root):")
    for n in names:
        print(f"  {n} -> {resolved[n]}")

    # ---- 2. list adapters, label them ----
    for gname in names:
        folder = resolved[gname]
        files = list(src.iter_files(folder, recursive=True))
        if not files:
            raise LabelError(f"folder '{folder}' is empty")
        paths = {f.path for f in files}
        all_weights = [f for f in files if is_weight_file(f.path)]
        weights = [f for f in all_weights if weight_name_matches(f.path, args.weight_name)]
        if not weights:
            seen = sorted({f.path.rsplit("/", 1)[-1] for f in files})[:8]
            raise LabelError(f"folder '{folder}': no files matching '{args.weight_name}' "
                             f"(file names seen: {seen}); adjust --weight-name after checking s0 output")
        by_dir = {}
        for w in weights:
            by_dir.setdefault(w.path.rsplit("/", 1)[0], []).append(w)
        multi_weight_dirs += sum(len(v) > 1 for v in by_dir.values())
        hints = folder_hints(gname)
        for w in weights:
            if is_local and w.size < 4096:      # empty file or Git-LFS pointer text instead of real weights
                head = Path(src.fetch(w.path)).read_bytes()[:64]
                if w.size == 0 or head.startswith(b"version https://git-lfs"):
                    incomplete.append(w.path)
                    continue
            lab, why = resolver.label(w.path, folder)
            if lab is None:
                problems.setdefault(why, []).append(w.path); continue
            cfg = w.path.rsplit("/", 1)[0] + "/adapter_config.json"
            rows.append({"item_id": w.path, "label": lab, "y": int(lab == "backdoor"), "group": gname,
                         "label_source": why if why != "ok" else resolver.mode, "data_tag": data_tag,
                         "label_mapping_status": spec.get("mapping_status", "n/a"),
                         "relpath": relpath_in_folder(w.path, folder), "size_bytes": w.size,
                         "config_path": cfg if cfg in paths else "", **hints})
    if problems.get("conflict"):
        raise LabelError(f"{len(problems['conflict'])} adapters match rules of BOTH classes, e.g. {problems['conflict'][:3]}")
    unl = [p for k, v in problems.items() for p in v]
    if any(k.startswith("unmapped_label_dir") for k in problems) and not args.drop_unlabeled:
        k = next(k for k in problems if k.startswith("unmapped_label_dir"))
        raise LabelError(f"label folder not in the mapping ({k}), e.g. {problems[k][:2]}. Add it to 'map' in the rules "
                         f"file only if its meaning is known, or exclude it explicitly with --drop-unlabeled.")
    if unl and not args.drop_unlabeled:
        raise LabelError(f"{len(unl)} adapters are unlabeled by the rules, e.g. {unl[:3]} "
                         f"(fix the rules, or pass --drop-unlabeled to exclude them explicitly)")
    avail = pd.DataFrame(rows)
    if avail.empty:
        raise LabelError("no labeled adapters found" + (f" ({len(incomplete)} files were empty/Git-LFS pointers - "
                         "the real weights were not downloaded)" if incomplete else ""))
    report = verify_manifest(avail, resolver, args.min_per_class, args.allow_contradictions)
    report["unlabeled_dropped"] = len(unl)
    report.update({"data_tag": data_tag, "revision": revision, "weight_name": args.weight_name,
                   "label_mode": resolver.mode, "rules_file": Path(rules_path).name,
                   "mapping_status": spec.get("mapping_status"),
                   "folders": list(names), "folders_resolved": resolved, "root_prefix": prefix, "seed": args.seed,
                   "local_root_effective": str(src.root) if is_local else None,
                   "skipped_incomplete_files": len(incomplete)})
    report["available_total"] = len(avail)
    report["dirs_with_multiple_weight_files"] = multi_weight_dirs
    if incomplete:
        report["warnings"].append(f"{len(incomplete)} weight files are empty or Git-LFS pointers (not real weights) and "
                                  f"were skipped, e.g. {incomplete[:2]}")
    if multi_weight_dirs:
        report["warnings"].append("some directories hold several weight files (e.g. checkpoints); each is a separate item")
    parts = []
    for (_, _), g in avail.groupby(["group", "label"]):
        parts.append(g.sample(n=min(len(g), args.max_per_class_per_folder), random_state=int(rng.integers(1 << 31))))
    man = pd.concat(parts).sort_values("item_id").reset_index(drop=True)
    man.to_csv(work / "manifest.csv", index=False)
    report["selected"] = man["label"].value_counts().to_dict()
    report["selected_bytes_gb"] = round(float(man["size_bytes"].sum()) / 1e9, 3)
    (work / "manifest_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
except LabelError as e:
    print(f"LABEL VERIFICATION FAILED: {e}", file=sys.stderr); sys.exit(2)

print("data_tag:", data_tag, "| label rules:", rules_path)
if report.get("mapping_status") != "verified":
    print(f"LABEL MAPPING STATUS: {str(report.get('mapping_status')).upper()} - {report.get('label_dir_map')}; not confirmed against "
          "authoritative PADBench metadata.")
print("available labeled adapters:", report["available_total"], "| selected:", report["selected"],
      f"| size of selected files ~{report['selected_bytes_gb']} GB")
print("\nper-folder class counts (available):")
print(pd.DataFrame(report["per_group"]).T.to_string())
for w in report["warnings"]:
    print("WARNING:", w)
print("\nwrote", work / "manifest.csv")
