#!/usr/bin/env python3
"""Stage 3 - leakage-safe split, baseline classifiers, validation-chosen thresholds, held-out evaluation.

Split: whole config folders (the `group` column) are assigned to exactly one of train/val/test, so related
adapters (same base model / dataset / attack / rank) never straddle splits. Test therefore measures transfer
to unseen configurations, which is harder (and more honest) than a random split.
"""
import argparse, json, sys, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from c1.common import is_real_tag
from c1.features import METRICS, feature_names
from c1.model_utils import (_logit, binary_metrics, choose_thresholds, final_scores, raw_scores, verdict_from_score)

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work")
ap.add_argument("--model", choices=["auto", "lr", "rf"], default="auto", help="auto = best validation AUROC")
ap.add_argument("--n-folds", type=int, default=5)
ap.add_argument("--test-fold", type=int, default=0)
ap.add_argument("--val-fold", type=int, default=1)
ap.add_argument("--group-by", choices=["config", "family"], default="config",
                help="config = hold out whole folders (same recipe+rank stays together); family = hold out whole "
                     "base-model/dataset/attack families (stricter; needs >= 3 families)")
ap.add_argument("--no-within-config-fallback", action="store_true",
                help="stop (instead of falling back to an adapter-level split) when fewer than 3 independent folders exist")
ap.add_argument("--test-groups", nargs="*", help="explicit split units (folder names, or family names with --group-by family) for test")
ap.add_argument("--val-groups", nargs="*", help="explicit split units for validation")
ap.add_argument("--min-per-class-eval", type=int, default=5, help="each of val/test needs this many per class")
ap.add_argument("--fpr-block", type=float, default=0.01)
ap.add_argument("--recall-flag", type=float, default=0.95)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

work = Path(args.work); mdir = work / "model"; mdir.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(work / "manifest.csv")
mrep = json.loads((work / "manifest_report.json").read_text(encoding="utf-8"))
tags = set(man["data_tag"])
if len(tags) != 1:
    sys.exit(f"manifest mixes data sources {sorted(tags)}; refusing")
data_tag = tags.pop()
REAL = is_real_tag(data_tag)
MAPPING = mrep.get("mapping_status")
print("=" * 78)
print(("REAL PADBench data (%s), revision %s" % (data_tag, mrep.get("revision"))) if REAL else
      f"!! NOT REAL PADBENCH: data_tag='{data_tag}'. Results below are a code-path check, NOT dataset results. !!")
print("label mapping:", mrep.get("label_dir_map") or mrep.get("label_mode"), "| status:", str(MAPPING).upper())
if MAPPING != "verified":
    print("!! LABELS ARE PROVISIONAL: the benign/backdoor meaning of the label folders is not confirmed against "
          "authoritative PADBench metadata. If it is inverted, every verdict below is inverted. !!")
print("evidence:", mrep.get("label_evidence", "n/a"))
print("=" * 78)
feat = pd.read_csv(work / "features" / "features.csv").drop_duplicates("item_id", keep="last")
df = man.merge(feat, on="item_id", how="left")
bad = df[df.status != "ok"]
print(f"manifest={len(man)}  usable={int((df.status=='ok').sum())}  missing/error={len(bad)}")
if len(bad):
    print(bad[["item_id", "status", "error"]].head(8).to_string(index=False))
df = df[df.status == "ok"].reset_index(drop=True).copy()

# ---- group split (no config folder in more than one split) ----
unit = "family" if args.group_by == "family" else "group"
groups = df[unit].nunique()
n_families = df["family"].nunique()
split_mode = "group"
if groups < 3:
    if args.no_within_config_fallback:
        sys.exit(f"need >= 3 independent {unit} units for a train/val/test group split (have {groups}); add folders/"
                 f"families in Stage 1. With fewer, no held-out metric is computed.")
    split_mode = "within_config"
    print(f"\n!! Only {groups} independent {unit} unit(s) available -> FALLBACK to a stratified adapter-level split "
          f"inside the same configuration (60/20/20, seed {args.seed}). Adapters of the same recipe appear in train and "
          f"test, so this measures in-distribution detectability ONLY, not generalisation. !!")
if split_mode == "group":
    k = min(args.n_folds, groups)
    if args.test_fold >= k or args.val_fold >= k or args.test_fold == args.val_fold:
        sys.exit(f"--test-fold/--val-fold must be distinct and < n_folds ({k})")
    fold = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed)
    for f, (_, idx) in enumerate(sgkf.split(df, df["y"], df[unit])):
        fold[idx] = f
    df["split"] = np.where(fold == args.test_fold, "test", np.where(fold == args.val_fold, "val", "train"))
    if args.test_groups or args.val_groups:
        tg, vg = set(args.test_groups or []), set(args.val_groups or [])
        unknown = (tg | vg) - set(df[unit])
        if unknown or (tg & vg):
            sys.exit(f"bad --test-groups/--val-groups (unknown={sorted(unknown)}, overlap={sorted(tg & vg)}); have {sorted(df[unit].unique())}")
        df["split"] = np.where(df[unit].isin(tg), "test", np.where(df[unit].isin(vg), "val", "train"))
    assert all(df.groupby(unit)["split"].nunique() == 1), f"LEAKAGE: a {unit} spans several splits"
    assert all(df.groupby("group")["split"].nunique() == 1), "LEAKAGE: a config folder spans several splits"
else:
    from sklearn.model_selection import train_test_split
    try:
        idx = np.arange(len(df))
        tr_i, rest_i = train_test_split(idx, test_size=0.4, stratify=df["y"], random_state=args.seed)
        va_i, te_i = train_test_split(rest_i, test_size=0.5, stratify=df["y"].to_numpy()[rest_i], random_state=args.seed)
    except ValueError as e:
        sys.exit(f"too few adapters per class for a 60/20/20 split ({e}). Add more adapters (stage 1 "
                 f"--max-per-class-per-folder) or more folders.")
    df["split"] = "train"
    df.loc[va_i, "split"] = "val"
    df.loc[te_i, "split"] = "test"
tab = df.groupby(["split", "group", "label"]).size().unstack(fill_value=0)
print("\nsplit assignment (%s):\n" % ("whole folders" if split_mode == "group" else "adapters within one config"), tab.to_string())
for sp in ("val", "test"):
    c = df[df.split == sp]["label"].value_counts()
    if min(c.get("benign", 0), c.get("backdoor", 0)) < args.min_per_class_eval:
        sys.exit(f"{sp} split lacks both classes (>= {args.min_per_class_eval} each): {c.to_dict()}. "
                 f"Add folders, change --test-fold/--val-fold, or lower --min-per-class-eval.")
if df[df.split == "train"]["y"].nunique() < 2:
    sys.exit("train split has a single class")

limitations = []
if not REAL:
    limitations.append(f"DATA SOURCE '{data_tag}' IS NOT REAL PADBENCH - nothing here is a dataset result.")
if MAPPING != "verified":
    limitations.append("LABEL MAPPING IS PROVISIONAL (not confirmed against authoritative PADBench metadata): if label0/"
                       "label1 are swapped, all metrics and verdicts are inverted.")
if split_mode == "within_config":
    limitations.append("Adapter-level split inside ONE configuration: train and test share base model, dataset, attack "
                       "and rank. Metrics show in-distribution detectability only and will overstate real-world performance.")
if n_families == 1:
    limitations.append("All adapters come from ONE family (same base model/dataset/attack/PEFT method; they differ "
                       "only in rank and per-adapter training subset). Holding out folders tests rank transfer only; "
                       "generalisation to other datasets, attacks, base models or PEFT methods is NOT tested.")
if groups < 5:
    limitations.append(f"Only {groups} independent {unit} units: validation and test are each a single folder/family "
                       f"(or two), so metrics and thresholds are high-variance and depend on which unit was held out.")
for sp_ in ("val", "test"):
    c_ = df[df.split == sp_]["label"].value_counts()
    if min(c_.get("benign", 0), c_.get("backdoor", 0)) < 30:
        limitations.append(f"{sp_} has {c_.get('benign', 0)} benign / {c_.get('backdoor', 0)} backdoor adapters (<30 per "
                           f"class): see the 95% intervals; do not over-read point estimates.")
cols_all = feature_names()
tr = df[df.split == "train"]
nz = [c for c in cols_all if np.isfinite(tr[c]).all() and tr[c].std() > 1e-12]
print(f"\nfeatures: {len(cols_all)} total, {len(nz)} non-constant on train")
X = lambda d: d[nz].to_numpy(dtype=np.float64)
ytr, va, te = tr["y"].to_numpy(), df[df.split == "val"], df[df.split == "test"]

# ---- candidates ----
cands = {
    "lr": Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))]),
    "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=args.seed),
}
val_auc = {}
for name, est in cands.items():
    est.fit(X(tr), ytr)
    val_auc[name] = roc_auc_score(va["y"], est.predict_proba(X(va))[:, 1])
print("validation AUROC:", {k_: round(v, 4) for k_, v in val_auc.items()})
chosen = args.model if args.model != "auto" else max(val_auc, key=lambda m: (val_auc[m], m == "lr"))
print("chosen model:", chosen)

bundle = {"version": 1, "model_name": chosen, "estimator": cands[chosen], "feature_cols": nz, "platt": None}
# Platt calibration on validation so the score behaves like a probability for the later fusion module
p_val = raw_scores(bundle, X(va))
platt = LogisticRegression(C=1.0).fit(_logit(p_val).reshape(-1, 1), va["y"])
bundle["platt"] = platt
s_val = final_scores(bundle, X(va))
th = choose_thresholds(va["y"], s_val, args.fpr_block, args.recall_flag)
bundle["thresholds"] = th
bundle["benign_mean"] = tr.loc[tr.y == 0, nz].mean().to_numpy()
bundle["benign_std"] = tr.loc[tr.y == 0, nz].std().to_numpy()

# per-layer benign reference stats (train benign only) for layer-level evidence in explanations
lay = pd.read_csv(work / "features" / "layer_features.csv")
ref_ids = set(tr.loc[tr.y == 0, "item_id"])
lr_ = lay[lay.item_id.isin(ref_ids)]
ls = {}
for (mod, layer), g in lr_.groupby(["module", "layer"]):
    for m in METRICS:
        ls[f"{mod}|{m}|{int(layer)}"] = [float(g[m].mean()), float(g[m].std(ddof=0))]
bundle["layer_stats"] = ls
bundle["meta"] = {"sklearn": sklearn.__version__, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "seed": args.seed, "train_groups": sorted(tr.group.unique()),
                  "val_groups": sorted(va.group.unique()), "test_groups": sorted(te.group.unique())}
joblib.dump(bundle, mdir / "c1_model.joblib")

# ---- scores for every row, all splits ----
df["score"] = final_scores(bundle, X(df))
df["verdict"] = [verdict_from_score(s, th) for s in df["score"]]
df[["item_id", "split", "group", "family", "label", "label_source", "data_tag", "y", "score", "verdict"]].to_csv(mdir / "scores.csv", index=False)
df[["item_id", "split", "group", "family", "label"]].to_csv(mdir / "split_assignment.csv", index=False)

def evaluate(d):
    y, s = d["y"].to_numpy(), d["score"].to_numpy()
    out = {"n": int(len(d)), "n_benign": int((y == 0).sum()), "n_backdoor": int((y == 1).sum()),
           "auroc": float(roc_auc_score(y, s)), "avg_precision": float(average_precision_score(y, s)),
           "block_only": binary_metrics(y, d["verdict"] == "block"),
           "flag_or_block": binary_metrics(y, d["verdict"] != "pass"),
           "fixed_0.5": binary_metrics(y, s >= 0.5),
           "verdict_counts": d["verdict"].value_counts().to_dict()}
    out["per_group"] = {g: {"n": int(len(x)), "recall_flag_or_block": binary_metrics(x["y"], x["verdict"] != "pass")["recall"],
                            "fpr_flag_or_block": binary_metrics(x["y"], x["verdict"] != "pass")["fpr"]}
                        for g, x in d.groupby("group")}
    return out

metrics = {"data_tag": data_tag, "revision": mrep.get("revision"), "group_by": args.group_by,
           "n_split_units": int(groups), "n_families": int(n_families), "limitations": limitations,
           "label_mapping": mrep.get("label_dir_map") or mrep.get("label_mode"),
           "label_evidence": mrep.get("label_evidence"), "label_mapping_status": MAPPING,
           "split_mode": split_mode, "model": chosen, "val_auroc_by_model": val_auc, "thresholds": th,
           "train": evaluate(df[df.split == "train"]), "val": evaluate(df[df.split == "val"]),
           "test": evaluate(df[df.split == "test"])}
(mdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=lambda o: None if o != o else float(o)), encoding="utf-8")

print(f"\nthresholds (from validation only): FLAG >= {th['t_flag']:.4f}   BLOCK >= {th['t_block']:.4f}")
print(f"  val FPR @block={th['val_fpr_at_block']:.3f}  val recall @flag={th['val_recall_at_flag']:.3f}")
for split in ("val", "test"):
    m = metrics[split]
    print(f"\n=== {split.upper()}  n={m['n']} (benign {m['n_benign']}, backdoor {m['n_backdoor']})  "
          f"AUROC={m['auroc']:.4f}  AP={m['avg_precision']:.4f}  verdicts={m['verdict_counts']}")
    for pol in ("block_only", "flag_or_block"):
        b = m[pol]
        print(f"  [{pol:13s}] confusion  TN={b['tn']:4d} FP={b['fp']:4d} | FN={b['fn']:4d} TP={b['tp']:4d}   "
              f"precision={b['precision']:.3f} recall={b['recall']:.3f} F1={b['f1']:.3f} FPR={b['fpr']:.3f}")
        print(f"{'':17s} 95% CI: recall [{b['recall_ci95'][0]:.3f}, {b['recall_ci95'][1]:.3f}]  FPR [{b['fpr_ci95'][0]:.3f}, {b['fpr_ci95'][1]:.3f}]")
print(f"\n(train AUROC {metrics['train']['auroc']:.4f} - compare with val/test to spot overfitting)")
print("\nLIMITATIONS OF THIS EVALUATION:" if limitations else "\n(no automatic limitations raised)")
for l_ in limitations:
    print("  -", l_)
print("saved:", mdir / "c1_model.joblib", mdir / "metrics.json", mdir / "scores.csv")
