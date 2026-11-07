#!/usr/bin/env python3
"""
ML-B evaluation: ablation (tabular vs tabular+graph, per configuration), validation-set threshold choice,
hard rules -> block, GNN go/no-go note, and the trained bundle used by scorer.py.

    python ml_b_eval.py --out out_sim          (needs ML-A's out folder: events.csv, labels.csv, splits.csv)

Protocol: tasks are grouped into folds (ML-A's splits.csv). For each TEST fold f, fold (f+1)%n is the VALIDATION fold,
the other folds train. Thresholds are chosen on VALIDATION benign agents only; TEST shows what the chosen threshold really does.
"""
import argparse, json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import mla_pipeline as P
from metrics_extra import classification_metrics
import graph_features as G

PRIMARY_K = P.PRIMARY_K
FPR = P.FPR_TARGET
HARD_RULE_COLS = ["g_pingpong", "g_n_novel_edges", "a_n_err"]    # justified on validation: see rule_limits
MODEL = "LR_cal"      # logistic regression first; RF_cal is run alongside as the comparison
N_BOOT = 1000


def fit_model(name, X, y, groups):
    """Same recipe as ML-A's fit_predict but returns the fitted model (for the saved bundle)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold
    if len(np.unique(y)) < 2:
        class DummyModel:
            def predict_proba(self, X):
                p = np.zeros((len(X), 2))
                p[:, 0] = 1.0
                return p
        return DummyModel()
    if name.endswith("_cal"):
        splits = list(GroupKFold(n_splits=3).split(X, y, groups))
        if y.sum() >= 6 and all(len(np.unique(y[a])) == 2 and y[b].sum() >= 1 for a, b in splits):
            return CalibratedClassifierCV(estimator=P.base_model(name), method="sigmoid", cv=splits).fit(X, y)
    return P.base_model(name).fit(X, y)


def prepare(out):
    ev = pd.read_csv(f"{out}/events.csv")
    lab = pd.read_csv(f"{out}/labels.csv")
    spl = pd.read_csv(f"{out}/splits.csv")
    meta = lab.merge(spl[["run_id", "task_fold"]], on="run_id").rename(columns={"agent_invoked_attack_tool": "y"})
    harmless_runs = set(lab.loc[lab.attack_type == P.HARMLESS, "run_id"])
    return ev, lab, meta, harmless_runs


def dataset_for_fold(ev, roster, meta, harmless_runs, K, train_tasks):
    """tabular + graph features for every agent-run; novelty reference uses TRAIN tasks only."""
    tab = P.build_features(ev, roster, K)
    eidx = G.benign_edge_index(ev, harmless_runs)
    gr = G.graph_features(ev, roster, K, edge_index=eidx, allowed_tasks=train_tasks)
    d = tab.merge(gr, on=["run_id", "agent"]).merge(meta, on=["run_id", "agent"])
    return d[~((d["y"] == 1) & (d["first_attack_idx"] < K))].reset_index(drop=True)


def run_cv(ev, roster, meta, harmless_runs, K, cols_sets, models):
    """returns long table of test+validation predictions and per-fold thresholds."""
    folds = sorted(meta["task_fold"].unique())
    rows, thr_rows = [], []
    task_fold = meta.drop_duplicates("task_id").set_index("task_id")["task_fold"]
    for f in folds:
        v = folds[(folds.index(f) + 1) % len(folds)]
        train_tasks = set(task_fold[~task_fold.isin([f, v])].index)
        d = dataset_for_fold(ev, roster, meta, harmless_runs, K, train_tasks)
        tr, va, te = d[~d.task_fold.isin([f, v])], d[d.task_fold == v], d[d.task_fold == f]
        for sname, cols in cols_sets.items():
            ridx = [cols.index(c) for c in P.RULE_COLS]
            for m in models:
                Xev = np.vstack([va[cols].to_numpy(float), te[cols].to_numpy(float)])
                s = P.fit_predict(m, tr[cols].to_numpy(float), tr["y"].to_numpy(int), tr["task_id"].to_numpy(),
                                  Xev, ridx)
                sv, st = s[:len(va)], s[len(va):]
                thr = P.threshold_at_fpr(va["y"].to_numpy(), sv)
                thr_rows.append(dict(K=K, fold=f, val_fold=v, features=sname, model=m, threshold=thr,
                                     val_fpr=float((sv[va.y.to_numpy() == 0] > thr).mean()),
                                     test_fpr=float((st[te.y.to_numpy() == 0] > thr).mean()),
                                     test_recall=float((st[te.y.to_numpy() == 1] > thr).mean()),
                                     n_val=len(va), n_test=len(te)))
                part = te[["run_id", "agent", "task_id", "config", "scenario", "attack_type", "y"]].copy()
                part = part.assign(K=K, fold=f, features=sname, model=m, score=st, thr=thr)
                rows.append(part)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(thr_rows)


def summarize(pred, group_cols):
    out = []
    for key, g in pred.groupby(group_cols + ["K", "features", "model"]):
        y, s, thr = g["y"].to_numpy(), g["score"].to_numpy(), g["thr"].to_numpy()
        pos, neg = (y == 1), (y == 0)
        cm = classification_metrics(y, s, thr)          # threshold chosen on validation tasks, applied to unseen test tasks
        out.append(dict(zip(group_cols + ["K", "features", "model"], key),
                        n=len(g), n_pos=int(pos.sum()), accuracy=cm["accuracy"], precision=cm["precision"],
                        recall=cm["recall"], f1=cm["f1"], roc_auc=cm["roc_auc"], pr_auc=cm["pr_auc"],
                        fpr_at_val_thr=round(float((s[neg] > thr[neg]).mean()), 4) if neg.any() else np.nan))
    return pd.DataFrame(out)


def paired_bootstrap(pred, K, model, seed=0):
    """PR-AUC(tab+graph) - PR-AUC(tab) with tasks resampled; returns mean, 95% CI, share of resamples > 0."""
    a = pred[(pred.K == K) & (pred.model == model) & (pred.features == "tab")].set_index(["run_id", "agent"])
    b = pred[(pred.K == K) & (pred.model == model) & (pred.features == "tab+graph")].set_index(["run_id", "agent"])
    b = b.loc[a.index]
    tasks = a["task_id"].unique()
    by_task = {t: np.where(a["task_id"].to_numpy() == t)[0] for t in tasks}
    rng = np.random.default_rng(seed)
    ya, sa, sb = a["y"].to_numpy(), a["score"].to_numpy(), b["score"].to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        ix = np.concatenate([by_task[t] for t in rng.choice(tasks, len(tasks))])
        if ya[ix].sum() == 0:
            continue
        diffs.append(average_precision_score(ya[ix], sb[ix]) - average_precision_score(ya[ix], sa[ix]))
    d = np.array(diffs)
    if len(d) == 0:
        return dict(mean_gain=0.0, ci_low=0.0, ci_high=0.0, p_gain_positive=0.0)
    return dict(mean_gain=float(d.mean()), ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)),
                p_gain_positive=float((d > 0).mean()))


def gnn_note(boot, n_runs, n_configs, path, primary_k=PRIMARY_K):
    ci_low = boot.get("ci_low", 0.0)
    mean_gain = boot.get("mean_gain", 0.0)
    p_gain_pos = boot.get("p_gain_positive", 0.0)
    go = (ci_low is not None and not np.isnan(ci_low) and ci_low > 0 and mean_gain >= 0.03 and n_runs >= 2000)
    lines = ["# GNN go / no-go note (ML-B with ML-A)\n",
             "Rule agreed in advance: consider a GNN only if graph features CLEARLY help (PR-AUC gain >= 0.03 with the 95% task-bootstrap "
             "interval above 0) AND there is enough graph data (>= 2000 runs).\n",
             f"- PR-AUC gain from adding graph features (K={primary_k}, {MODEL}): {mean_gain:+.4f} "
             f"(95% CI {ci_low:+.4f} to {boot.get('ci_high', 0.0):+.4f}; gain > 0 in {p_gain_pos:.0%} of resamples)",
             f"- runs available: {n_runs} (needed >= 2000)",
             f"- configurations present in the data: {n_configs}\n",
             f"## Decision: **{'GO' if go else 'NO-GO'}**\n",
             ("Graph features clearly help and data is sufficient; a GNN is worth trying." if go else
              "Keep hand-built graph features plus the tree/linear models. Revisit when (a) real logs from several configurations "
              "(Swarm, CrewAI, orchestrator) exist and (b) the ablation shows a clear gain.")]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return go


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    a = ap.parse_known_args(argv)[0]
    info = json.load(open(f"{a.out}/run_info.json", encoding="utf-8"))
    if info.get("status") == "insufficient_data_no_training":
        raise SystemExit(f"{a.out}: too few examples for supervised ML-B training. Graph-feature extraction is still tested "
                         "(python -m unittest discover tests). See README 'TAMAS Ground-Truth Labels and Limitations'.")
    ev, lab, meta, harmless_runs = prepare(a.out)
    if not harmless_runs:
        print("[note] no Harmless runs in these logs: the 'edge never seen in harmless runs' features stay 0 (not informative).")
    roster = lab[["run_id", "agent"]]
    G.selfcheck(ev, roster)
    print("[1] graph self-check passed (future events do not move features; hand-made ping-pong log gives expected values)")

    tab_cols = P.FEATURE_COLS
    sets = {"tab": tab_cols, "tab+graph": tab_cols + G.GRAPH_COLS}
    k_list = info.get("K_LIST", P.K_LIST)
    primary_k = info.get("primary_K", PRIMARY_K)
    preds, thrs = [], []
    for K in k_list:
        p, t = run_cv(ev, roster, meta, harmless_runs, K, sets, ["LR_cal", "RF_cal"])
        preds.append(p), thrs.append(t)
        print(f"[2] K={K} cross-validation done")
    pred = pd.concat(preds, ignore_index=True)
    thr = pd.concat(thrs, ignore_index=True)
    pred.to_csv(f"{a.out}/mlb_predictions.csv", index=False)

    abl = summarize(pred, ["config"])
    abl.to_csv(f"{a.out}/mlb_ablation_by_config.csv", index=False)
    thr.to_csv(f"{a.out}/mlb_threshold_by_fold.csv", index=False)
    boot = paired_bootstrap(pred, primary_k, MODEL)
    boot_rows = [dict(K=K, model=MODEL, **paired_bootstrap(pred, K, MODEL)) for K in k_list]
    pd.DataFrame(boot_rows).to_csv(f"{a.out}/mlb_ablation_bootstrap.csv", index=False)

    # ---- hard rules, justified on VALIDATION benign agents (limit = max benign value seen on validation)
    K = primary_k
    folds = sorted(meta["task_fold"].unique())
    task_fold = meta.drop_duplicates("task_id").set_index("task_id")["task_fold"]
    rule_rows, limits_by_fold = [], []
    for f in folds:
        v = folds[(folds.index(f) + 1) % len(folds)]
        train_tasks = set(task_fold[~task_fold.isin([f, v])].index)
        d = dataset_for_fold(ev, roster, meta, harmless_runs, K, train_tasks)
        va, te = d[d.task_fold == v], d[d.task_fold == f]
        for c in HARD_RULE_COLS:
            lim = float(va.loc[va.y == 0, c].max())
            fire = te[c] > lim
            rule_rows.append(dict(rule=f"{c} > {lim:g}", fold=f, limit=lim, test_fpr=float(fire[te.y == 0].mean()),
                                  test_recall=float(fire[te.y == 1].mean()), n_fired=int(fire.sum())))
    rules = pd.DataFrame(rule_rows)
    rules.to_csv(f"{a.out}/mlb_hard_rules.csv", index=False)

    # ---- final bundle for scorer.py: trained on ALL tasks at the primary K, threshold from pooled validation scores
    all_tasks = set(meta["task_id"])
    d = dataset_for_fold(ev, roster, meta, harmless_runs, K, all_tasks)
    ci_low = boot.get("ci_low")
    chosen = "tab+graph" if (ci_low is not None and not np.isnan(ci_low) and ci_low > 0) else "tab"   # keep the simpler set unless the gain is clear
    cols = sets[chosen]
    model = fit_model(MODEL, d[cols].to_numpy(float), d["y"].to_numpy(int), d["task_id"].to_numpy())
    vthr = thr[(thr.K == K) & (thr.features == chosen) & (thr.model == MODEL)]
    threshold = float(vthr["threshold"].median()) if len(vthr) and not np.isnan(vthr["threshold"].median()) else 0.5
    ben = d[d.y == 0][cols]
    limits = {c: float(d.loc[d.y == 0, c].quantile(1.0)) for c in HARD_RULE_COLS}
    joblib.dump(dict(model=model, cols=cols, K=K, threshold=threshold, feature_set=chosen, rule_limits=limits,
                     edge_index=G.benign_edge_index(ev, harmless_runs), train_tasks=sorted(all_tasks),
                     benign_mean=ben.mean().to_dict(), benign_std=(ben.std() + 1e-9).to_dict()),
                f"{a.out}/mlb_bundle.joblib")

    go = gnn_note(boot, n_runs=int(meta.run_id.nunique()), n_configs=int(meta.config.nunique()),
                  path=f"{a.out}/mlb_gnn_decision.md", primary_k=primary_k)

    # ---- reports
    main_tbl = abl[abl.K == primary_k].sort_values(["config", "model", "features"])
    lines = ["# ML-B: ablation, thresholds, hard rules\n", P.banner(info["data_source"], info["label_mode"]),
             f"## Ablation by configuration (K={primary_k}; threshold chosen on validation tasks at {FPR:.0%} benign FPR)\n",
             main_tbl.to_markdown(index=False),
             "\nRound Robin's fixed order may add little, so this is tested, not assumed. Only the configurations present in the logs appear.\n",
             f"## Does adding graph features help? paired bootstrap over tasks ({MODEL})\n",
             pd.DataFrame(boot_rows).round(4).to_markdown(index=False),
             "\n## Threshold report: threshold chosen on validation, then checked on unseen test tasks\n",
             thr[(thr.K == primary_k) & (thr.model == MODEL)].round(4).to_markdown(index=False),
             "\n## Hard rules (block): limit = largest benign value on validation tasks\n",
             rules.round(4).to_markdown(index=False),
             f"\n## Final bundle\nfeature set: **{chosen}**, model {MODEL}, K={K}, flag threshold (risk) {threshold:.4f} "
             f"(median of per-fold validation thresholds), hard-rule limits {json.dumps({k: round(v, 3) for k, v in limits.items()})}.\n",
             f"GNN decision: **{'GO' if go else 'NO-GO'}** (see mlb_gnn_decision.md)"]
    open(f"{a.out}/mlb_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
