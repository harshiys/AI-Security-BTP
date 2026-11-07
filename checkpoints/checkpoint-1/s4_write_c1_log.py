#!/usr/bin/env python3
"""Stage 4 - score adapters, map to PASS/FLAG/BLOCK, write schema-validated C1 log records.

Each log line has EXACTLY the shared-schema fields (checkpoint_id,item_id,verdict,score,timestamp).
Explanations go to a separate sidecar file keyed by item_id (+ the same timestamp), because the schema
sets additionalProperties:false.
"""
import argparse, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from c1.common import CHECKPOINT_ID, is_real_tag
from c1.features import feature_names
from c1.model_utils import Explainer, final_scores, verdict_from_score

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work")
ap.add_argument("--schema", default="shared_schema_format.json", help="the team's shared schema, unmodified")
ap.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
ap.add_argument("--out-dir", default=None, help="default: <work>/c1_log")
ap.add_argument("--allow-fallback-validator", action="store_true",
                help="ONLY if jsonschema is not installable: use a minimal built-in checker (reported in output)")
ap.add_argument("--pubkey", default="scanner_public.pem", help="path to scanner_public.pem")
ap.add_argument("--classifier-record", default=None, help="path to classifier_record.json (default: <work>/model/classifier_record.json)")
ap.add_argument("--skip-model-verification", action="store_true", help="bypass classifier seal check (not recommended)")
args = ap.parse_args()

work = Path(args.work); out = Path(args.out_dir or work / "c1_log"); out.mkdir(parents=True, exist_ok=True)
schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
REQUIRED = set(schema["required"])

# ---------- validation ----------
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
try:
    import jsonschema
    _v = jsonschema.Draft7Validator(schema, format_checker=jsonschema.FormatChecker())
    VALIDATOR = f"jsonschema {jsonschema.__version__}"
    def schema_errors(rec): return [e.message for e in _v.iter_errors(rec)]
except ImportError:
    if not args.allow_fallback_validator:
        sys.exit("jsonschema is not installed (pip install jsonschema). Refusing to write unvalidated logs.")
    VALIDATOR = "BUILT-IN FALLBACK (jsonschema missing)"
    def schema_errors(rec):
        errs, props = [], schema["properties"]
        errs += [f"missing {k}" for k in REQUIRED if k not in rec]
        if schema.get("additionalProperties") is False:
            errs += [f"additional property {k}" for k in rec if k not in props]
        for k, v in rec.items():
            p = props.get(k, {})
            if p.get("type") == "string" and not isinstance(v, str): errs.append(f"{k} not string")
            if p.get("type") == "number" and (isinstance(v, bool) or not isinstance(v, (int, float))): errs.append(f"{k} not number")
            if "enum" in p and v not in p["enum"]: errs.append(f"{k} not in enum")
            if "minimum" in p and isinstance(v, (int, float)) and v < p["minimum"]: errs.append(f"{k} below minimum")
            if "maximum" in p and isinstance(v, (int, float)) and v > p["maximum"]: errs.append(f"{k} above maximum")
        return errs

def validate(rec):
    errs = schema_errors(rec)
    if set(rec) != REQUIRED: errs.append(f"keys {sorted(rec)} != required {sorted(REQUIRED)}")
    ts = rec.get("timestamp")
    if isinstance(ts, str):
        if not _TS.match(ts): errs.append("timestamp is not ISO8601 date-time")
        else:
            try:
                if datetime.fromisoformat(ts).utcoffset() is None: errs.append("timestamp has no UTC offset")
            except ValueError as e: errs.append(f"timestamp unparseable: {e}")
    return errs

# ---------- provenance: real PADBench vs synthetic ----------
mrep = json.loads((work / "manifest_report.json").read_text(encoding="utf-8"))
metrics_p = work / "model" / "metrics.json"
metrics = json.loads(metrics_p.read_text(encoding="utf-8")) if metrics_p.exists() else {}
_man_tags = set(pd.read_csv(work / "manifest.csv")["data_tag"])
if len(_man_tags) != 1: sys.exit(f"manifest mixes data sources {sorted(_man_tags)}; refusing")
DATA_TAG = _man_tags.pop()
IS_REAL = is_real_tag(DATA_TAG)
MAPPING = mrep.get("mapping_status")
print("=" * 78)
print(("RUN TYPE: REAL PADBench (%s), revision %s" % (DATA_TAG, mrep.get("revision"))) if IS_REAL else
      f"RUN TYPE: !! NOT REAL PADBENCH (data_tag='{DATA_TAG}') - log records are a code-path check, not dataset results !!")
if MAPPING != "verified":
    print(f"LABEL MAPPING: {str(MAPPING).upper()} ({mrep.get('label_dir_map')}) - not confirmed against authoritative "
          "PADBench metadata; 'ground_truth' in the explanations file inherits this.")
print("=" * 78)

# ---------- score ----------
classifier_record = args.classifier_record or str(work / "model" / "classifier_record.json")
if args.skip_model_verification:
    bundle = joblib.load(work / "model" / "c1_model.joblib")
else:
    from verified_load import load_verified
    bundle = load_verified(str(work / "model" / "c1_model.joblib"),
                           record_path=classifier_record,
                           pubkey=args.pubkey)
th, cols = bundle["thresholds"], bundle["feature_cols"]
man = pd.read_csv(work / "manifest.csv")
feat = pd.read_csv(work / "features" / "features.csv").drop_duplicates("item_id", keep="last")
split = pd.read_csv(work / "model" / "split_assignment.csv")[["item_id", "split"]]
df = man.merge(feat, on="item_id").merge(split, on="item_id")
df = df[(df.status == "ok") & ((df.split == args.split) | (args.split == "all"))].reset_index(drop=True).copy()
if df.empty: sys.exit(f"no feature rows for split '{args.split}'")
X = df[cols].to_numpy(dtype=np.float64)
scores = final_scores(bundle, X)
lay = pd.read_csv(work / "features" / "layer_features.csv")
lay = lay[lay.item_id.isin(set(df.item_id))]
explainer = Explainer(bundle)

def layer_evidence(item_id, top_feats, k=3):
    g, ev, seen = lay[lay.item_id == item_id], [], set()
    for tf in top_feats:
        mod, metric, _ = tf["feature"].split("__")
        if (mod, metric) in seen: continue
        seen.add((mod, metric))
        best = None
        for r in g[g.module == mod].itertuples():
            ref = bundle["layer_stats"].get(f"{mod}|{metric}|{int(r.layer)}")
            if not ref: continue
            val = float(getattr(r, metric)); z = (val - ref[0]) / ref[1] if ref[1] > 0 else 0.0
            if best is None or abs(z) > abs(best["z"]):
                best = {"module": mod, "layer": int(r.layer), "metric": metric, "value": val,
                        "benign_mean": ref[0], "benign_std": ref[1], "z": float(z)}
        if best: ev.append(best)
        if len(ev) >= k: break
    return ev

records, side, all_errs = [], [], []
for i, row in enumerate(df.itertuples()):
    s = float(np.clip(scores[i], 0.0, 1.0))
    rec = {"checkpoint_id": CHECKPOINT_ID, "item_id": row.item_id, "verdict": verdict_from_score(s, th),
           "score": s, "timestamp": datetime.now(timezone.utc).isoformat()}
    errs = validate(rec)
    if errs: all_errs.append((row.item_id, errs))
    top, method = explainer.top(X[i], 5)
    side.append({"item_id": rec["item_id"], "timestamp": rec["timestamp"], "verdict": rec["verdict"], "score": s,
                 "thresholds": {"flag": th["t_flag"], "block": th["t_block"]}, "model": bundle["model_name"],
                 "explanation_method": method, "top_risk_features": top,
                 "layer_evidence": layer_evidence(row.item_id, top),
                 "eval_only": {"ground_truth": row.label, "label_source": row.label_source, "group": row.group,
                               "split": row.split, "data_tag": row.data_tag,
                               "label_mapping_status": MAPPING}})
    records.append(rec)

if all_errs:
    print(f"SCHEMA VALIDATION FAILED for {len(all_errs)} records; nothing written. First: {all_errs[0]}", file=sys.stderr); sys.exit(2)
log_p, side_p = out / "ckpt1_log.jsonl", out / "ckpt1_explanations.jsonl"
log_p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
side_p.write_text("".join(json.dumps(r) + "\n" for r in side), encoding="utf-8")
vc = pd.Series([r["verdict"] for r in records]).value_counts().to_dict()
(out / "run_info.json").write_text(json.dumps({
    "run_type": "REAL_PADBENCH" if IS_REAL else "NOT_REAL_PADBENCH",
    "data_tag": DATA_TAG, "dataset_revision": mrep.get("revision"), "folders": mrep.get("folders"),
    "weight_name": mrep.get("weight_name"), "label_mode": mrep.get("label_mode"),
    "label_mapping": mrep.get("label_dir_map"), "label_mapping_status": MAPPING,
    "label_evidence": mrep.get("label_evidence"), "split_mode": metrics.get("split_mode"),
    "split_scored": args.split, "n_records": len(records), "model": bundle["model_name"],
    "thresholds": {"flag": th["t_flag"], "block": th["t_block"]},
    "group_by": metrics.get("group_by"), "test_metrics_file": str(metrics_p),
    "limitations": metrics.get("limitations", []),
    "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
(out / "validation_report.json").write_text(json.dumps(
    {"validator": VALIDATOR, "schema_file": str(args.schema), "records": len(records), "invalid": 0, "verdicts": vc,
     "data_tag": DATA_TAG, "label_mapping_status": MAPPING}, indent=2), encoding="utf-8")
print(f"run_info -> {out / 'run_info.json'}")
print(f"validator: {VALIDATOR}\nwrote {len(records)} schema-valid records -> {log_p}\nverdicts: {vc}\nexplanations -> {side_p}")
print("\nfirst 3 log records:"); [print(" ", json.dumps(r)) for r in records[:3]]
