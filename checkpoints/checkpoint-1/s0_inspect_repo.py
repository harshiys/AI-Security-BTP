#!/usr/bin/env python3
"""Stage 0 - inspect the REAL repo structure and look for label evidence. Lists only; downloads tiny metadata files.

Why it samples per sub-folder: a recursive listing is returned in path order, so with a cap the first sub-folder
(e.g. label0) can fill the cap and a later one (label1) is never seen. Each immediate sub-folder is therefore
sampled separately.

Output: console report, <work>/inspect_report.json, and (only if label evidence is found)
<work>/label_rules.proposed.json - a PROPOSAL/TEMPLATE. Stage 1 refuses it until a human has filled in the
mapping + evidence, removed its 'status' key and saved it as <work>/label_rules.json.
"""
import argparse, collections, itertools, json, re, sys
from pathlib import Path

from c1.common import (DEFAULT_FOLDERS, FolderResolutionError, describe_tree, discover_config_folders,
                       is_meta_file, is_weight_file, make_source, norm_name, resolve_folders)
from c1.labels import BACKDOOR_HINTS, BENIGN_HINTS, UNVERIFIED_MSG, tokens

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["hf", "local"], default="hf")
ap.add_argument("--local-root", help="folder containing the config folders (default: <work>/hf_files)")
ap.add_argument("--work", default="work")
ap.add_argument("--revision", default=None, help="pin a dataset commit sha (default: current)")
ap.add_argument("--folders", nargs="*", default=None,
                help="config folder names (default: hf = default rank folders; local = every folder that has weights)")
ap.add_argument("--root-prefix", default=None, help="only accept config folders whose parent is this path")
ap.add_argument("--max-entries", type=int, default=400, help="cap on files listed per sub-folder")
ap.add_argument("--max-meta-files", type=int, default=3)
args = ap.parse_args()

LABEL_DIR = re.compile(r"^label\d+$")
work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
names_arg = [norm_name(x) for x in args.folders] if args.folders else None
src = make_source(args.source, args.local_root, work, args.revision, names_arg)
print("source:", args.source, "| revision:", getattr(src, "revision", None),
      "| local root:" if args.source == "local" else "", getattr(src, "root", "") if args.source == "local" else "")

print("== repo top level (folders) ==")
top = list(src.iter_folders(""))
print(f"{len(top)} folders; first 40: {top[:40]}")
try:
    if names_arg is None and args.source == "local":
        full = discover_config_folders(src)
        args.folders, prefix = sorted(full), ""
        missing = [] if full else ["(none found)"]
    else:
        args.folders = names_arg or list(DEFAULT_FOLDERS)
        prefix, full, missing = resolve_folders(src, args.folders, args.root_prefix)
except FolderResolutionError as e:
    sys.exit(f"DATASET DISCOVERY FAILED: {e}")
print("config folders found:", {n: full[n] for n in args.folders if n in full})
if missing:
    print(f"!! requested/expected folders NOT found: {missing}")
    if args.source == "local":
        print("   what is actually under the local root:\n" + describe_tree(src))
    else:
        print(f"   top-level folders in the repo: {top[:40]}")
    print("   -> pass --local-root (parent of the config folders) and/or --folders with the real names.")
report = {"revision": getattr(src, "revision", None), "root_prefix": prefix, "folders": {}}
all_tokens, label_dirs, weight_names = collections.Counter(), collections.Counter(), collections.Counter()
any_capped = False
for name in args.folders:
    if name in missing:
        continue
    folder = full[name]
    children = list(src.iter_folders(folder))
    entries, capped = [], False
    if children:
        entries += list(itertools.islice(src.iter_files(folder, recursive=False), 100))
        for ch in children:
            got = list(itertools.islice(src.iter_files(ch, recursive=True), args.max_entries))
            capped |= len(got) >= args.max_entries
            entries += got
    else:
        entries = list(itertools.islice(src.iter_files(folder, recursive=True), args.max_entries))
        capped = len(entries) >= args.max_entries
    any_capped |= capped
    rels = [e.path[len(folder) + 1:] for e in entries]
    weights = [e for e in entries if is_weight_file(e.path)]
    for e in weights:
        weight_names[e.path.rsplit("/", 1)[-1]] += 1
    ext = collections.Counter(Path(r).suffix.lower() or "<none>" for r in rels)
    names = collections.Counter(r.rsplit("/", 1)[-1] for r in rels)
    toks = collections.Counter(t for r in rels for t in tokens(r))
    all_tokens.update(toks)
    per_depth = collections.defaultdict(collections.Counter)
    for r in rels:
        for d, comp in enumerate(r.split("/")[:-1]):
            per_depth[d][comp] += 1
            if LABEL_DIR.match(comp):
                label_dirs[comp] += 1
    folder_labels = collections.Counter()
    for e in weights:
        for comp in e.path[len(folder) + 1:].split("/")[:-1]:
            if LABEL_DIR.match(comp):
                folder_labels[comp] += 1
    print(f"\n== {folder} ==  sampled={len(entries)} weight_files={len(weights)} capped={capped}")
    print("  immediate sub-folders:", children[:12])
    print("  extensions:", dict(ext))
    print("  most common file names:", names.most_common(6))
    print("  directory names by depth (top 6 each):")
    for d in sorted(per_depth):
        shown = [f"{k}({v})" for k, v in per_depth[d].most_common(6)]
        print(f"    depth {d}: {shown}{' ...' if len(per_depth[d]) > 6 else ''}")
    print("  weight files per label-like folder (sampled):", dict(folder_labels) or "none found")
    sizes = [e.size for e in weights]
    if sizes:
        print(f"  weight size MB: min={min(sizes)/1e6:.1f} median={sorted(sizes)[len(sizes)//2]/1e6:.1f} max={max(sizes)/1e6:.1f}")
    print("  first paths:"); [print("    ", r) for r in rels[:8]]
    shown = 0
    for e in entries:
        if is_meta_file(e.path) and e.size < 1_000_000 and shown < args.max_meta_files:
            shown += 1
            try:
                txt = Path(src.fetch(e.path)).read_text(errors="replace", encoding="utf-8").splitlines()[:15]
                print(f"  -- metadata file {e.path} (first lines) --"); [print("     ", l[:160]) for l in txt]
            except Exception as ex:  # noqa: BLE001
                print("  could not read", e.path, ex)
    report["folders"][name] = {"sampled": len(entries), "capped": capped, "weight_files": len(weights),
                                 "children": children, "extensions": dict(ext),
                                 "weight_files_per_label_dir": dict(folder_labels),
                                 "dir_names_by_depth": {d: dict(c.most_common(10)) for d, c in per_depth.items()},
                                 "first_paths": rels[:25]}

report["weight_file_names"] = dict(weight_names)
report["label_like_dirs"] = dict(label_dirs)
(work / "inspect_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n== weight file names seen ==", dict(weight_names))
if "adapter_model.safetensors" not in weight_names:
    print("!! no 'adapter_model.safetensors' seen in the sample; Stage 1 filters on that name by default "
          "(change with --weight-name) - check the names above.")

b_found = sorted(set(all_tokens) & BENIGN_HINTS); k_found = sorted(set(all_tokens) & BACKDOOR_HINTS)
print("\n== label evidence ==")
print("numeric label folders:", dict(label_dirs) or "none")
print("benign-like words in paths:", b_found, "| backdoor-like words in paths:", k_found)
if label_dirs:
    prop = {"status": "PROPOSED/UNVERIFIED - fill 'map' and 'evidence', delete this key, save as label_rules.json",
            "mode": "label_dir_map", "dir_pattern": r"^label\d+$",
            "map": {d: None for d in sorted(label_dirs)},
            "evidence": ""}
    (work / "label_rules.proposed.json").write_text(json.dumps(prop, indent=2), encoding="utf-8")
    print(f"\nFound label folders {sorted(label_dirs)}. {UNVERIFIED_MSG}")
    print(f"Wrote a TEMPLATE with null mappings: {work / 'label_rules.proposed.json'}  (Stage 1 will refuse it as-is.)")
elif b_found and k_found:
    mk = lambda ts: r"(?i)(^|[^a-z])(" + "|".join(ts) + r")([^a-z]|$)"
    prop = {"status": "PROPOSED - review against inspect_report.json, then delete this key and save as label_rules.json",
            "mode": "path_regex",
            "rules": [{"pattern": mk(b_found), "label": "benign"}, {"pattern": mk(k_found), "label": "backdoor"}]}
    (work / "label_rules.proposed.json").write_text(json.dumps(prop, indent=2), encoding="utf-8")
    print(f"wrote {work / 'label_rules.proposed.json'} (a proposal only)")
else:
    print("No label evidence in file paths. Check metadata files above (mode=metadata_file) or the PEFTGuard "
          "documentation/code, then write label_rules.json by hand. Do NOT guess.")
if any_capped:
    print("note: some sub-folders were capped; counts are samples only (Stage 1 lists everything).")
