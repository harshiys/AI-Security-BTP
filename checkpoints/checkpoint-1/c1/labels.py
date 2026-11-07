"""Verified label parsing for PADBench adapters.

Nothing is assumed about how PADBench encodes benign vs backdoored. Labels come ONLY from a rules file
that a human has reviewed (work/label_rules.json), in one of two modes:

  path_regex     - regexes applied to the adapter path *relative to its config folder*
                   (the config folder name itself, e.g. '..._toxic_backdoors_hard_...', describes the
                   attack dataset, not the label, so it is excluded on purpose).
  metadata_file  - a CSV/JSON file in the repo mapping an adapter key to a label value.
  label_dir_map  - categorical folders such as label0/label1 inside each config folder. Their meaning is NOT
                   documented in the PADBench dataset card or the PEFTGuard README, so the mapping must be
                   written out explicitly *together with the evidence for it* (e.g. a quote/link from the
                   authors' train.py, the paper, or the authors' reply). Null/missing mappings are refused.

The parser fails closed: conflicting rules, unmatched adapters, contradictory path tokens, or a missing
class all raise LabelError instead of silently guessing.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

LABELS = ("benign", "backdoor")
BACKDOOR_HINTS = {"backdoor", "backdoored", "poison", "poisoned", "trojan", "trojaned", "malicious", "triggered", "bd"}
BENIGN_HINTS = {"benign", "clean", "normal", "safe", "uncompromised"}


class LabelError(Exception):
    pass


MAPPING_STATUSES = ("provisional", "verified", "synthetic_fixture")


UNVERIFIED_MSG = (
    "The meaning of numeric label folders (label0 / label1) is NOT documented in the PADBench dataset card or the "
    "PEFTGuard README, and this pipeline refuses to guess it. Confirm it from an authoritative source (the data "
    "loading code in the official PEFTGuard train.py, the paper appendix, or the authors via the GitHub issue "
    "tracker), then fill `map` and `evidence` in <work>/label_rules.json.")


def tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z]+", s.lower()) if t}


def load_rules(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise LabelError(f"{p} not found. Run s0_inspect_repo.py, review work/label_rules.proposed.json, "
                         f"then save the reviewed rules as {p}.")
    spec = json.loads(p.read_text(encoding="utf-8"))
    if str(spec.get("status", "")).upper().startswith("PROPOSED"):
        raise LabelError(f"{p} still has a 'status: PROPOSED' key. Review the rules against the inspect report "
                         f"and delete that key to confirm them.")
    mode = spec.get("mode")
    if mode == "path_regex":
        rules = spec.get("rules") or []
        if not rules:
            raise LabelError("path_regex mode needs a non-empty 'rules' list")
        for r in rules:
            if r.get("label") not in LABELS:
                raise LabelError(f"rule label must be one of {LABELS}: {r}")
            re.compile(r["pattern"])
    elif mode == "label_dir_map":
        m = spec.get("map")
        if not isinstance(m, dict) or not m:
            raise LabelError("label_dir_map mode needs a non-empty 'map' such as {\"label0\": \"...\", \"label1\": \"...\"}")
        bad = {k: v for k, v in m.items() if v not in LABELS}
        if bad:
            raise LabelError(f"label_dir_map entries {bad} are not set to 'benign' or 'backdoor'. {UNVERIFIED_MSG}")
        if set(m.values()) != set(LABELS):
            raise LabelError(f"label_dir_map must map to BOTH classes {LABELS}; got {sorted(set(m.values()))}")
        if len(str(spec.get("evidence", "")).strip()) < 15:
            raise LabelError("label_dir_map needs a written 'evidence' (>= 15 chars) saying where the mapping was "
                             "verified (link/quote/author reply). " + UNVERIFIED_MSG)
        re.compile(spec.get("dir_pattern", r"^label\d+$"))
        if spec.get("mapping_status") not in MAPPING_STATUSES:
            raise LabelError(f"label_dir_map needs 'mapping_status' set to one of {MAPPING_STATUSES}. Use 'provisional' "
                             f"until the mapping is confirmed against authoritative PADBench documentation.")
    elif mode == "metadata_file":
        for k in ("file", "path_column", "label_column", "label_map", "path_match"):
            if k not in spec:
                raise LabelError(f"metadata_file mode needs '{k}'")
        if spec["path_match"] not in ("exact", "relpath", "parent_dir_name", "parent_relpath"):
            raise LabelError("path_match must be exact|relpath|parent_dir_name|parent_relpath")
        if not set(spec["label_map"].values()) <= set(LABELS):
            raise LabelError(f"label_map values must be in {LABELS}")
    else:
        raise LabelError("mode must be 'path_regex', 'label_dir_map' or 'metadata_file'")
    return spec


def relpath_in_folder(path: str, folder: str) -> str:
    return path[len(folder) + 1:] if path.startswith(folder + "/") else path


class LabelResolver:
    def __init__(self, spec: dict, source=None):
        self.spec, self.mode = spec, spec["mode"]
        self.rule_hits: Counter = Counter()
        self.dir_hits: Counter = Counter()
        self.meta: dict[str, str] = {}
        if self.mode == "metadata_file":
            local = source.fetch(spec["file"])
            if str(local).lower().endswith(".json"):
                obj = json.loads(Path(local).read_text(encoding="utf-8"))
                df = pd.DataFrame(obj) if isinstance(obj, list) else pd.DataFrame(
                    [{spec["path_column"]: k, spec["label_column"]: v} for k, v in obj.items()])
            else:
                df = pd.read_csv(local, dtype=str, sep=None, engine="python")
            for c in (spec["path_column"], spec["label_column"]):
                if c not in df.columns:
                    raise LabelError(f"column '{c}' not in metadata file (columns: {list(df.columns)})")
            raw_vals = {str(v) for v in df[spec["label_column"]]}
            unmapped = raw_vals - set(map(str, spec["label_map"]))
            if unmapped:
                raise LabelError(f"label values {sorted(unmapped)} missing from label_map")
            for k, v in zip(df[spec["path_column"]], df[spec["label_column"]]):
                self.meta[str(k)] = spec["label_map"][str(v)]

    def label(self, path: str, folder: str) -> tuple[str | None, str]:
        rel = relpath_in_folder(path, folder)
        if self.mode == "label_dir_map":
            pat = re.compile(self.spec.get("dir_pattern", r"^label\d+$"))
            hits = [c for c in rel.split("/")[:-1] if pat.fullmatch(c)]
            if not hits:
                for c in rel.split("/")[:-1]:
                    m = re.search(r"(?:^|[_\-.])(label\d+)(?:[_\-.]|$)", c, re.I)
                    if m and (m.group(1) in self.spec.get("map", {})):
                        hits.append(m.group(1))
            if not hits:
                # fallback: label token inside the file name, e.g. label1_0007.safetensors
                m = re.search(r"(?:^|[_\-.])(label\d+)(?:[_\-.]|$)", rel.rsplit("/", 1)[-1], re.I)
                if m and pat.fullmatch(m.group(1)):
                    lab = self.spec["map"].get(m.group(1))
                    if lab is None:
                        return None, f"unmapped_label_dir:{m.group(1)}"
                    self.dir_hits[m.group(1)] += 1
                    return lab, f"label_file:{m.group(1)}"
                return None, "no_match"
            if len(set(hits)) > 1:
                return None, "conflict"
            lab = self.spec["map"].get(hits[0])
            if lab is None:
                return None, f"unmapped_label_dir:{hits[0]}"
            self.dir_hits[hits[0]] += 1
            return lab, f"label_dir:{hits[0]}"
        if self.mode == "path_regex":
            labs, hits = set(), []
            for i, r in enumerate(self.spec["rules"]):
                if re.search(r["pattern"], rel):
                    labs.add(r["label"]); hits.append(i)
            for i in hits:
                self.rule_hits[i] += 1
            if len(labs) == 1:
                return labs.pop(), "ok"
            return None, ("conflict" if labs else "no_match")
        pm = self.spec["path_match"]
        key = {"exact": path, "relpath": rel,
               "parent_dir_name": rel.rsplit("/", 2)[-2] if rel.count("/") >= 1 else "",
               "parent_relpath": rel.rsplit("/", 1)[0] if "/" in rel else ""}[pm]
        lab = self.meta.get(key)
        return (lab, "ok") if lab else (None, "metadata_missing")


def verify_manifest(df: pd.DataFrame, resolver: LabelResolver, min_per_class: int = 10,
                    allow_contradictions: bool = False) -> dict:
    """df has item_id,label,group,relpath. Raises LabelError on hard failures; returns a report."""
    rep: dict = {"warnings": []}
    counts = df["label"].value_counts().to_dict()
    rep["class_counts"] = counts
    for lab in LABELS:
        if counts.get(lab, 0) < min_per_class:
            raise LabelError(f"class '{lab}' has {counts.get(lab, 0)} adapters (< {min_per_class}); counts found: {counts}. "
                             f"Not enough data for a meaningful split. Add more adapters of that class, or lower "
                             f"--min-per-class if you accept a very small evaluation.")
    contra = []
    for item, lab, rel in zip(df["item_id"], df["label"], df["relpath"]):
        t = tokens(rel)
        if (lab == "benign" and t & BACKDOOR_HINTS) or (lab == "backdoor" and t & BENIGN_HINTS):
            contra.append(item)
    rep["contradictory_tokens"] = len(contra)
    if contra and not allow_contradictions:
        raise LabelError(f"{len(contra)} adapters have path tokens contradicting their label, e.g. {contra[:3]}")
    per = df.groupby(["group", "label"]).size().unstack(fill_value=0)
    for lab in LABELS:
        if lab not in per.columns:
            per[lab] = 0
    rep["per_group"] = per[list(LABELS)].to_dict(orient="index")
    single = [g for g, r in rep["per_group"].items() if min(r.values()) == 0]
    if single:
        rep["warnings"].append(f"single-class folders (group-wise splits will be unusable for these): {single}")
    if resolver.mode == "label_dir_map":
        dead_dirs = [d for d in resolver.spec["map"] if resolver.dir_hits[d] == 0]
        if dead_dirs:
            rep["warnings"].append(f"mapped label folders with no adapters: {dead_dirs}")
        rep["label_dir_map"] = resolver.spec["map"]
        rep["label_evidence"] = resolver.spec["evidence"]
        rep["mapping_status"] = resolver.spec["mapping_status"]
    if resolver.mode == "path_regex":
        dead = [i for i in range(len(resolver.spec["rules"])) if resolver.rule_hits[i] == 0]
        if dead:
            rep["warnings"].append(f"rules that matched nothing: {dead}")
    return rep
