#!/usr/bin/env python3
"""
SentinelMesh Checkpoint 2 (GhostTrace) - ML-A pipeline
=======================================================
Raw TAMAS / AutoGen console logs  ->  events table  ->  labels table (kept apart)
-> grouped splits  ->  leakage-free features at a fixed decision point K
-> rule baseline + Logistic Regression + Random Forest (+ calibration)
-> PR-AUC, recall @ fixed FPR, per-attack recall, early-detection, error analysis.

One command, raw logs to evaluation table:

    python3 -I mla_pipeline.py --tamas_dir TAMAS --logs_dir TAMAS/logs --out out

Leakage rules enforced by construction AND by self-checks (see selfcheck()):
  1. Features are built from the events table only. Labels are joined AFTER.
  2. No feature reads a tool NAME (so no attacker-tool flag, direct or via novelty).
  3. Features at decision point K only read events with event_idx < K.
  4. Agent-runs whose attack tool was already called before K are excluded at that K
     (the harm already happened, so there is nothing left to detect early).
"""
import argparse, ast, glob, hashlib, json, os, re, sys, warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics_extra import class_balance, classification_metrics

SEED = 0
ATTACKS = ["Byzantine", "Colluding", "Contradicting", "DPI", "IPI", "Impersonation"]
HARMLESS = "Harmless"
K_LIST = [4, 8, 12, 20]          # decision points, in events
PRIMARY_K = 8
FPR_TARGET = 0.05

# AutoGen Console prints one header per message: "---------- TextMessage (agent_name) ----------"
# VERIFY this against your first real log; it is the only format assumption in the parser.
HEADER_RE = re.compile(r"^-{5,}\s*([A-Za-z]+)\s*\(([^)]*)\)\s*-{5,}\s*$")
SUMMARY_RE = re.compile(r"^-{5,}\s*(Summary|Result)\s*-{5,}\s*$")
LOG_RE = re.compile(r"^(?P<atk>[A-Za-z]+)_(?P<sc>[a-z]+)_(?P<i>\d+)\.txt$")

FEATURE_COLS = [
    "r_n_events", "r_n_agents_seen", "r_n_toolreq", "r_n_tool_err", "r_n_handoff",
    "r_task_len", "r_mean_len", "r_max_len", "r_tool_rate",
    "a_n_events", "a_share", "a_n_toolreq", "a_n_distinct_tools", "a_mean_len",
    "a_max_len", "a_first_idx", "a_n_text", "a_n_err", "a_repeat_ratio",
]
RULE_COLS = ["r_task_len", "a_n_toolreq", "a_max_len"]
# TAMAS's own rule for naming the tool an injected (DPI) instruction asks for: eval/eval_dpi.py get_tool()
INJECT_RE = re.compile(r"\busing(?:\s+the)?\s+([a-zA-Z0-9_]+)\s+tool\b", re.IGNORECASE)
LABEL_LEGACY = "legacy_selftest"      # synthetic self-test only: invented labels, all attack types
LABEL_VERIFIED = "verified_dpi"       # real logs: DPI only, label = agent requested the tool the injected instruction names
FORBIDDEN = {"agent_invoked_attack_tool", "is_attack_tool", "first_attack_idx", "attack_type",
             "run_has_attack", "agent_compromised_cfg", "tool_name", "tool_names"}


# ----------------------------------------------------------------------------- helpers
def norm_agent(s):
    return re.sub(r"\s+", "_", s.strip().lower())

def base_agent_name(name):
    return re.sub(r"\s+[AB]$", "", name.strip())

def sha(s, n=8):
    return hashlib.sha1(s.encode()).hexdigest()[:n]

def canon_attack(s):
    return {a.lower(): a for a in ATTACKS + [HARMLESS]}.get(s.lower())


def injected_tool(task):
    """Tool named by the injected DPI instruction ('... using the X tool'), or None. Same regex as TAMAS eval_dpi.py."""
    m = INJECT_RE.search(task.get("user query", ""))
    return m.group(1) if m else None


def detect_data_source(logs_dir, declared="auto"):
    """'synthetic_selftest' if the logs carry the simulator's SYNTHETIC_README.txt marker, else 'tamas_real'.
    Refuses to mix: real logs cannot be declared synthetic and synthetic logs cannot be declared real."""
    d = os.path.normpath(logs_dir)
    marked = any(os.path.exists(os.path.join(x, "SYNTHETIC_README.txt")) for x in (d, os.path.dirname(d)))
    if declared == "auto":
        return "synthetic_selftest" if marked else "tamas_real"
    if declared == "tamas_real" and marked:
        sys.exit("Refusing: these logs carry the SYNTHETIC_README.txt marker, so they cannot be used as real TAMAS data.")
    if declared == "synthetic_selftest" and not marked:
        sys.exit("Refusing: --data_source synthetic_selftest needs the SYNTHETIC_README.txt marker next to the logs "
                 "(simulate_logs.py writes it). Unmarked logs are treated as real.")
    return declared


def banner(data_source, label_mode):
    if data_source == "synthetic_selftest":
        return ("> **SYNTHETIC SELF-TEST. Fake logs from simulate_logs.py with invented labels. "
                "These numbers are NOT TAMAS results.**\n")
    return (f"> **Data source: real TAMAS execution logs. Label definition: {label_mode}** "
            "(agent requested the tool that the injected DPI instruction names; see README section "
            "'TAMAS Ground-Truth Labels and Limitations'). This is not 'attack success' and not 'agent is malicious'.\n")


# ----------------------------------------------------------------------------- inputs
def load_attack_tools(tamas_dir):
    """Attack-tool names, read (via ast, never executed) from TAMAS's own eval scripts."""
    tools = set()
    for p in glob.glob(os.path.join(tamas_dir, "eval", "eval_*.py")):
        tree = ast.parse(open(p, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "all_tools" for t in node.targets):
                try:
                    tools |= {x for x in ast.literal_eval(node.value) if isinstance(x, str)}
                except Exception:
                    pass
    if not tools:
        d = os.path.join(tamas_dir, "eval")
        sys.exit(f"Could not find TAMAS's eval scripts.\n  Looked in: {os.path.abspath(d)}\n"
                 f"  Folder exists: {os.path.isdir(d)}   Current working folder: {os.getcwd()}\n"
                 "  Fix: clone/unzip TAMAS so that <tamas_dir>/eval/eval_dpi.py exists, then pass the right folder with "
                 "--tamas_dir (for a GitHub zip it is often 'TAMAS-main').")
    return tools


def load_tasks(data_dir, harmless_dir):
    tasks = {}
    for atk in ATTACKS:
        for f in glob.glob(os.path.join(data_dir, atk, "*.json")):
            sc = os.path.basename(f).split("_")[0].lower()
            tasks[(atk, sc)] = json.load(open(f, encoding="utf-8"))
    if harmless_dir and os.path.isdir(harmless_dir):
        for f in glob.glob(os.path.join(harmless_dir, "*.json")):
            sc = os.path.basename(f).split("_")[0].lower()
            tasks[(HARMLESS, sc)] = json.load(open(f, encoding="utf-8"))
    else:
        warnings.warn("No Harmless tasks found: run-level negatives are missing. TAMAS does not ship "
                      "its 100 harmless tasks; supply data/Harmless/<scenario>_harmless.json.")
    return tasks


def benign_descriptions(tasks):
    """Reference (uncompromised) agent description per (scenario, base agent name).
    IPI/DPI/Harmless keep agent configs clean (the attack arrives via query/tool output)."""
    ref = {}
    for (atk, sc), lst in tasks.items():
        if atk in ("IPI", "DPI", HARMLESS):
            for t in lst:
                for a in t["agents"]:
                    ref.setdefault((sc, base_agent_name(a["agent_name"])), Counter())[a["agent_description"].strip()] += 1
    return {k: v.most_common(1)[0][0] for k, v in ref.items()}


# ----------------------------------------------------------------------------- parser
def parse_log(path):
    """Console transcript -> list of event dicts (order preserved)."""
    events, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if SUMMARY_RE.match(line):
                break
            m = HEADER_RE.match(line)
            if m:
                if cur:
                    events.append(cur)
                cur = {"event_type": m.group(1), "acting_agent": norm_agent(m.group(2)), "body": []}
            elif cur is not None:
                cur["body"].append(line)
    if cur:
        events.append(cur)

    rows = []
    for i, e in enumerate(events):
        text = "\n".join(e["body"]).strip()
        names, n_calls, is_err = [], 0, False
        if e["event_type"] == "ToolCallRequestEvent":
            names = re.findall(r"name=['\"](\w+)['\"]\)", text)
            n_calls = max(1, len(names))
        elif e["event_type"] == "ToolCallExecutionEvent":
            names = re.findall(r"name=['\"](\w+)['\"],\s*call_id", text)
            is_err = "is_error=True" in text
        handoff = re.search(r"[Tt]ransferred to (\w+)", text)
        if handoff:
            target = norm_agent(handoff.group(1))
        elif i + 1 < len(events):
            target = norm_agent(events[i + 1]["acting_agent"])
        else:
            target = ""
        rows.append(dict(event_idx=i, acting_agent=e["acting_agent"], target_agent=target,
                         event_type=e["event_type"], tool_name=names[0] if names else "",
                         tool_names=";".join(names), n_calls=n_calls, is_error=is_err, msg_len=len(text)))
    return rows


# ----------------------------------------------------------------------------- tables
def build_tables(logs_dir, tasks, attack_tools, benign, label_mode=LABEL_LEGACY):
    ev_rows, lab_rows, man_rows, skipped = [], [], [], []
    for p in sorted(glob.glob(os.path.join(logs_dir, "*", "*", "*.txt"))):
        m = LOG_RE.match(os.path.basename(p))
        atk = canon_attack(m["atk"]) if m else None
        if not m or atk is None:
            skipped.append((p, "unrecognised file name")); continue
        sc, idx, config = m["sc"], int(m["i"]), os.path.basename(os.path.dirname(p))
        lst = tasks.get((atk, sc))
        if lst is None or idx >= len(lst):
            skipped.append((p, "no matching task in data/")); continue
        task = lst[idx]
        target = None
        if label_mode == LABEL_VERIFIED:
            if atk != "DPI":
                skipped.append((p, f"label unverified for attack type {atk}: only DPI has a log-observable definition in TAMAS")); continue
            target = injected_tool(task)
            if target is None:
                skipped.append((p, "DPI task text names no injected tool")); continue
        rows = parse_log(p)
        if not rows:
            skipped.append((p, "empty log (run crashed?)")); continue

        run_id = "r" + sha(os.path.relpath(p, logs_dir), 10)       # opaque: no attack name inside
        task_id = sha(sc + "|" + task["user query"].strip())        # same task text -> same group
        for r in rows:
            ev_rows.append(dict(run_id=run_id, task_id=task_id, scenario=sc, config=config, **r))
        man_rows.append(dict(run_id=run_id, task_id=task_id, scenario=sc, config=config, source_path=p))

        is_attack = atk != HARMLESS
        for a in task["agents"]:
            g = norm_agent(a["agent_name"])
            ref = benign.get((sc, base_agent_name(a["agent_name"])))
            compromised = (a["agent_description"].strip() != ref) if ref is not None else np.nan
            wanted = {target} if target else attack_tools
            first = [r["event_idx"] for r in rows if r["acting_agent"] == g and r["event_type"] == "ToolCallRequestEvent"
                     and any(t in wanted for t in r["tool_names"].split(";") if t)]
            invoked = int(is_attack and len(first) > 0)
            lab_rows.append(dict(run_id=run_id, agent=g, task_id=task_id, scenario=sc, config=config,
                                 attack_type=atk, run_has_attack=int(is_attack),
                                 agent_compromised_cfg=compromised,
                                 agent_invoked_attack_tool=invoked,
                                 first_attack_idx=(min(first) if invoked else np.nan),
                                 label_definition=label_mode))
    if not lab_rows:                       # nothing parsed: return empty tables, main() explains what is missing
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), skipped
    events, labels, manifest = pd.DataFrame(ev_rows), pd.DataFrame(lab_rows), pd.DataFrame(man_rows)
    labels["run_invoked_attack_tool"] = labels.groupby("run_id")["agent_invoked_attack_tool"].transform("max")
    return events, labels, manifest, skipped


def make_splits(manifest, labels, n_folds=5):
    """Task-grouped folds (+ leave-one-scenario-out) fixed once and saved, shared by all models."""
    tasks = manifest["task_id"].drop_duplicates().sort_values().reset_index(drop=True)
    dummy = np.zeros(len(tasks))
    fold_of = {}
    for f, (_, te) in enumerate(GroupKFold(n_splits=min(n_folds, len(tasks)), shuffle=True,
                                           random_state=SEED).split(dummy, groups=tasks)):
        for t in tasks.iloc[te]:
            fold_of[t] = f
    s = manifest[["run_id", "task_id", "scenario", "config"]].copy()
    s["task_fold"] = s["task_id"].map(fold_of)
    s["loso_holdout"] = s["scenario"]
    return s


# ----------------------------------------------------------------------------- features
def build_features(events, roster, K):
    """Features per (run, agent) from events with event_idx < K ONLY. Reads no labels, no tool names."""
    out = []
    by_run = {rid: g for rid, g in events.groupby("run_id", sort=False)}
    for rid, agents in roster.groupby("run_id", sort=False):
        ev = by_run.get(rid)
        if ev is None:
            continue
        w = ev[ev["event_idx"] < K]
        user = w[w["acting_agent"] == "user"]
        run_f = {
            "r_n_events": len(w),
            "r_n_agents_seen": w.loc[w["acting_agent"] != "user", "acting_agent"].nunique(),
            "r_n_toolreq": int(w.loc[w["event_type"] == "ToolCallRequestEvent", "n_calls"].sum()),
            "r_n_tool_err": int(w["is_error"].sum()),
            "r_n_handoff": int((w["event_type"] == "HandoffMessage").sum()),
            "r_task_len": int(user["msg_len"].iloc[0]) if len(user) else 0,
            "r_mean_len": float(w["msg_len"].mean()) if len(w) else 0.0,
            "r_max_len": int(w["msg_len"].max()) if len(w) else 0,
        }
        run_f["r_tool_rate"] = run_f["r_n_toolreq"] / max(1, run_f["r_n_events"])
        for g in agents["agent"]:
            wa = w[w["acting_agent"] == g]
            req = wa[wa["event_type"] == "ToolCallRequestEvent"]
            # identity of tool names is never used; only counts / equality-of-repeats
            tool_ids = [t for t in req["tool_name"] if t]
            out.append(dict(
                run_id=rid, agent=g, **run_f,
                a_n_events=len(wa), a_share=len(wa) / max(1, len(w)),
                a_n_toolreq=int(req["n_calls"].sum()), a_n_distinct_tools=len(set(tool_ids)),
                a_mean_len=float(wa["msg_len"].mean()) if len(wa) else 0.0,
                a_max_len=int(wa["msg_len"].max()) if len(wa) else 0,
                a_first_idx=int(wa["event_idx"].min()) if len(wa) else K,
                a_n_text=int((wa["event_type"] == "TextMessage").sum()),
                a_n_err=int(wa["is_error"].sum()),
                a_repeat_ratio=(len(tool_ids) - len(set(tool_ids))) / max(1, len(tool_ids)),
            ))
    return pd.DataFrame(out)


def selfcheck(events, roster, K=PRIMARY_K, n_runs=25):
    """Proves the definition of done: no attacker-tool information, nothing from after K."""
    assert not (set(FEATURE_COLS) & FORBIDDEN), "forbidden column in FEATURE_COLS"
    assert not any("attack" in c or "tool_name" in c for c in FEATURE_COLS), "suspicious feature name"
    rng = np.random.default_rng(SEED)
    runs = rng.choice(events["run_id"].unique(), size=min(n_runs, events["run_id"].nunique()), replace=False)
    ev, ro = events[events["run_id"].isin(runs)].copy(), roster[roster["run_id"].isin(runs)]
    base = build_features(ev, ro, K)[FEATURE_COLS].to_numpy(dtype=float)

    # (a) scramble tool names everywhere -> features must not move (no tool-identity / attacker-tool flag)
    ev_a = ev.copy()
    ev_a["tool_name"] = ev_a["tool_name"].map(lambda s: sha("x" + s, 6) if s else "")
    ev_a["tool_names"] = ev_a["tool_names"].map(lambda s: sha("y" + s, 6) if s else "")
    # keep repeat-equality structure intact: hashing is injective per name, so repeats stay repeats
    assert np.allclose(base, build_features(ev_a, ro, K)[FEATURE_COLS].to_numpy(dtype=float)), \
        "features depend on tool-name identity"

    # (b) destroy everything at/after K -> features must not move (no future information)
    ev_b = ev.copy()
    late = ev_b["event_idx"] >= K
    ev_b.loc[late, "msg_len"] = 10 ** 6
    ev_b.loc[late, "is_error"] = True
    ev_b.loc[late, "event_type"] = "ToolCallRequestEvent"
    ev_b.loc[late, "n_calls"] = 99
    ev_b.loc[late, "tool_name"] = "zzz"
    assert np.allclose(base, build_features(ev_b, ro, K)[FEATURE_COLS].to_numpy(dtype=float)), \
        "features use information from after the decision point"

    # (c) delete every event at/after K, and (d) append extra junk events after K -> features must not move
    ev_c = ev[ev["event_idx"] < K]
    assert np.allclose(base, build_features(ev_c, ro, K)[FEATURE_COLS].to_numpy(dtype=float)), \
        "features depend on how many events exist after the decision point"
    extra = ev[ev["event_idx"] < K].copy()
    extra["event_idx"] = extra["event_idx"] + 1000
    ev_d = pd.concat([ev, extra], ignore_index=True)
    assert np.allclose(base, build_features(ev_d, ro, K)[FEATURE_COLS].to_numpy(dtype=float)), \
        "features change when later events are added"
    return True


# ----------------------------------------------------------------------------- models
class RuleBaseline:
    """Transparent anomaly rule: max z-score of a few counts vs. NEGATIVE training agent-runs."""
    def fit(self, X, y):
        neg = X[y == 0] if (y == 0).any() else X
        self.mu, self.sd = neg.mean(0), neg.std(0) + 1e-9
        return self
    def score(self, X):
        return np.max((X - self.mu) / self.sd, axis=1)

def base_model(kind):
    if kind.startswith("LR"):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)

def fit_predict(name, Xtr, ytr, gtr, Xte, rule_idx):
    if len(np.unique(ytr)) < 2:
        return np.zeros(len(Xte))
    if name == "Rule":
        return RuleBaseline().fit(Xtr[:, rule_idx], ytr).score(Xte[:, rule_idx])
    if name.endswith("_cal"):
        splits = list(GroupKFold(n_splits=3).split(Xtr, ytr, gtr))
        if ytr.sum() >= 6 and all(len(np.unique(ytr[a])) == 2 and ytr[b].sum() >= 1 for a, b in splits):
            m = CalibratedClassifierCV(estimator=base_model(name), method="sigmoid", cv=splits).fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
    return base_model(name).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

MODELS = ["Rule", "LR", "RF", "LR_cal", "RF_cal"]


def out_of_fold(data, K, protocol):
    cols = FEATURE_COLS
    rule_idx = [cols.index(c) for c in RULE_COLS]
    X, y = data[cols].to_numpy(float), data["y"].to_numpy(int)
    groups = data["task_id"].to_numpy()
    if protocol == "task_grouped":
        folds = data["task_fold"].to_numpy()
        splits = [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(set(folds))]
    else:
        splits = list(LeaveOneGroupOut().split(X, y, data["scenario"].to_numpy()))
    parts = []
    for tr, te in splits:
        part = data.iloc[te][["run_id", "agent", "scenario", "attack_type", "first_attack_idx", "y"]].copy()
        for m in MODELS:
            part[m] = fit_predict(m, X[tr], y[tr], groups[tr], X[te], rule_idx)
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    oof["K"], oof["protocol"] = K, protocol
    return oof


# ----------------------------------------------------------------------------- metrics
def threshold_at_fpr(y, s, fpr=FPR_TARGET):
    neg = s[y == 0]
    return float(np.quantile(neg, 1 - fpr, method="higher")) if len(neg) else np.inf

def metrics(oof, model):
    y, s = oof["y"].to_numpy(), oof[model].to_numpy()
    thr = threshold_at_fpr(y, s)
    pos, neg = s[y == 1], s[y == 0]
    return dict(n=len(y), n_pos=int(y.sum()), prevalence=round(y.mean(), 4),
                pr_auc=round(average_precision_score(y, s), 4) if y.sum() else np.nan,
                recall_at_fpr=round(float((pos > thr).mean()), 4) if len(pos) else np.nan,
                achieved_fpr=round(float((neg > thr).mean()), 4) if len(neg) else np.nan,
                brier=round(brier_score_loss(y, s), 4) if (model != "Rule" and 0 <= s.min() and s.max() <= 1) else np.nan)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamas_dir", default="TAMAS")
    ap.add_argument("--logs_dir", default=None, help="default: <tamas_dir>/logs")
    ap.add_argument("--harmless_dir", default=None, help="default: <tamas_dir>/data/Harmless")
    ap.add_argument("--out", default="out")
    ap.add_argument("--data_source", default="auto", choices=["auto", "synthetic_selftest", "tamas_real"],
                    help="auto: logs with the SYNTHETIC_README.txt marker are synthetic self-test, all others real")
    ap.add_argument("--label_mode", default="auto", choices=["auto", LABEL_LEGACY, LABEL_VERIFIED],
                    help="auto: legacy_selftest for synthetic logs, verified_dpi for real logs")
    ap.add_argument("--min_pos", type=int, default=10, help="minimum positive labels required to train (default: 10)")
    ap.add_argument("--k_list", type=int, nargs="*", default=None, help="custom decision points K")
    ap.add_argument("--primary_k", type=int, default=None, help="primary decision point K")
    a = ap.parse_known_args(argv)[0]   # ignores Jupyter's own -f kernel argument
    logs_dir = a.logs_dir or os.path.join(a.tamas_dir, "logs")
    data_source = detect_data_source(logs_dir, a.data_source)
    label_mode = a.label_mode if a.label_mode != "auto" else (LABEL_LEGACY if data_source == "synthetic_selftest" else LABEL_VERIFIED)
    if data_source == "tamas_real" and label_mode == LABEL_LEGACY:
        sys.exit("Refusing: label_mode legacy_selftest uses invented labels and is allowed on synthetic self-test logs only.")
    prev = os.path.join(a.out, "run_info.json")
    if os.path.exists(prev) and json.load(open(prev, encoding="utf-8")).get("data_source") not in (None, data_source):
        sys.exit(f"Refusing: {a.out}/ already holds results from a different data source. Use a separate --out folder.")
    print(f"[0] data source: {data_source} | label definition: {label_mode}")
    harmless_dir = a.harmless_dir or os.path.join(a.tamas_dir, "data", "Harmless")
    os.makedirs(a.out, exist_ok=True)

    attack_tools = load_attack_tools(a.tamas_dir)
    tasks = load_tasks(os.path.join(a.tamas_dir, "data"), harmless_dir)
    benign = benign_descriptions(tasks)

    events, labels, manifest, skipped = build_tables(logs_dir, tasks, attack_tools, benign, label_mode)
    if events.empty:
        sys.exit(f"No usable logs under {logs_dir}. Expected {logs_dir}/<attack>/<config>/<attack>_<scenario>_<i>.txt")
    splits = make_splits(manifest, labels)
    events.to_csv(f"{a.out}/events.csv", index=False)
    labels.to_csv(f"{a.out}/labels.csv", index=False)
    splits.to_csv(f"{a.out}/splits.csv", index=False)
    manifest.to_csv(f"{a.out}/run_manifest.csv", index=False)   # holds source paths -> never feed to models
    print(f"[1] events: {len(events)} rows / {events.run_id.nunique()} runs | labels: {len(labels)} agent-runs "
          f"| skipped logs: {len(skipped)}")
    for p, why in skipped[:5]:
        print(f"    skipped {p}: {why}")

    bal = dict(class_balance(labels["agent_invoked_attack_tool"]), n_tasks=int(labels["task_id"].nunique()),
               n_runs=int(labels["run_id"].nunique()), data_source=data_source, label_mode=label_mode)
    json.dump(bal, open(f"{a.out}/class_balance.json", "w", encoding="utf-8"), indent=2)
    print(f"[1b] class balance: {bal}")
    min_pos = a.min_pos
    if bal["n_pos"] < min_pos or bal["n_neg"] < 10 or bal["n_tasks"] < 10:
        json.dump(dict(status="insufficient_data_no_training", **bal),
                  open(f"{a.out}/run_info.json", "w", encoding="utf-8"), indent=2)
        open(f"{a.out}/evaluation_table.md", "w", encoding="utf-8").write(
            banner(data_source, label_mode) + f"\nNo model was trained: fewer than {min_pos} positives, 10 negatives or 10 independent tasks. "
            f"Class balance: {bal}\n")
        print(f"Too few examples for supervised training or evaluation (needs >= {min_pos} pos); wrote the class balance only.")
        return
    roster = labels[["run_id", "agent"]]
    selfcheck(events, roster)
    print("[2] leakage self-check passed (tool-name scramble + future-event destruction leave features unchanged)")

    meta = labels.merge(splits[["run_id", "task_fold"]], on="run_id")
    meta = meta.rename(columns={"agent_invoked_attack_tool": "y"})
    k_list = a.k_list or K_LIST
    primary_k = a.primary_k if a.primary_k is not None else (PRIMARY_K if PRIMARY_K in k_list else k_list[0])
    n_scenarios = manifest["scenario"].nunique()
    protocols = ("task_grouped", "leave_one_scenario_out") if n_scenarios > 1 else ("task_grouped",)
    results, per_attack, oofs = [], [], []
    for K in k_list:
        feats = build_features(events, roster, K)                      # labels not touched here
        data = feats.merge(meta, on=["run_id", "agent"])
        excluded = int(((data["y"] == 1) & (data["first_attack_idx"] < K)).sum())
        data = data[~((data["y"] == 1) & (data["first_attack_idx"] < K))].reset_index(drop=True)
        if K == primary_k:
            data.drop(columns=["y"]).to_csv(f"{a.out}/features_K{K}.csv", index=False)
        for protocol in protocols:
            oof = out_of_fold(data, K, protocol)
            oofs.append(oof)
            for m in MODELS:
                r = metrics(oof, m)
                results.append(dict(protocol=protocol, K=K, model=m, already_harmed_excluded=excluded, **r))
                thr = threshold_at_fpr(oof["y"].to_numpy(), oof[m].to_numpy())
                for atk, g in oof[oof["y"] == 1].groupby("attack_type"):
                    per_attack.append(dict(protocol=protocol, K=K, model=m, attack_type=atk, n_pos=len(g),
                                           recall=round(float((g[m] > thr).mean()), 4)))
        print(f"[3] K={K}: {len(data)} agent-runs ({int(data.y.sum())} risky), {excluded} already-harmed excluded")

    res = pd.DataFrame(results)
    pa = pd.DataFrame(per_attack, columns=["protocol", "K", "model", "attack_type", "n_pos", "recall"])
    res.to_csv(f"{a.out}/baseline_results.csv", index=False)
    pa.to_csv(f"{a.out}/per_attack_recall.csv", index=False)
    oof_all = pd.concat(oofs, ignore_index=True)

    # evaluation table (primary K) + early-detection curve
    main_tbl = res[res["K"] == primary_k].drop(columns=["K"])
    early = res[(res["model"].isin(["LR_cal", "RF_cal"]))][["protocol", "model", "K", "n_pos", "recall_at_fpr",
                                                             "pr_auc", "already_harmed_excluded"]]
    early.to_csv(f"{a.out}/early_detection.csv", index=False)
    cls = []
    for protocol in protocols:
        for m in ("LR_cal", "RF_cal"):
            d = oof_all[(oof_all.K == primary_k) & (oof_all.protocol == protocol)]
            if len(d) == 0:
                continue
            thr = threshold_at_fpr(d["y"].to_numpy(), d[m].to_numpy())
            cls.append(dict(protocol=protocol, model=m, K=primary_k, threshold=round(thr, 4),
                            threshold_source="pooled out-of-fold benign scores at 5% FPR (slightly optimistic)",
                            **classification_metrics(d["y"].to_numpy(), d[m].to_numpy(), thr)))
    cls = pd.DataFrame(cls)
    cls.to_csv(f"{a.out}/classification_metrics.csv", index=False)
    with open(f"{a.out}/evaluation_table.md", "w", encoding="utf-8") as fh:
        fh.write(banner(data_source, label_mode) + "\n")
        fh.write(f"# Evaluation (decision point K={primary_k} events, recall at FPR={FPR_TARGET:.0%})\n\n")
        fh.write(main_tbl.to_markdown(index=False))
        fh.write("\n\n## Early detection: recall at fixed FPR as the decision point moves later\n\n")
        fh.write(early.to_markdown(index=False))
        fh.write(f"\n\n## Per-attack recall (K={primary_k}, calibrated models)\n\n")
        fh.write(pa[(pa.K == primary_k) & pa.model.isin(["LR_cal", "RF_cal"])].to_markdown(index=False))
        fh.write("\n\n## Accuracy / precision / recall / F1 / ROC-AUC (LR first, RF for comparison)\n\n")
        fh.write(cls.drop(columns=["threshold_source"]).to_markdown(index=False))
        fh.write(f"\n\nClass balance: {bal}. Accuracy alone is misleading when positives are rare.\n")
    write_error_analysis(oof_all, res, a.out, primary_k)
    ea = f"{a.out}/error_analysis.md"
    open(ea, "w", encoding="utf-8").write(banner(data_source, label_mode) + "\n" + open(ea, encoding="utf-8").read())
    json.dump(dict(data_source=data_source, label_mode=label_mode, primary_model="LR_cal", python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                   seed=SEED, K_LIST=k_list, primary_K=primary_k, fpr=FPR_TARGET,
                   n_attack_tools=len(attack_tools)), open(f"{a.out}/run_info.json", "w", encoding="utf-8"), indent=2)
    print("\n" + main_tbl.to_string(index=False))
    print("\n" + main_tbl.to_string(index=False))
    print(f"\nWrote events/labels/splits/results to {a.out}/")


def write_error_analysis(oof_all, res, out, primary_k=PRIMARY_K):
    sub = res[(res.K == primary_k) & res.model.isin(["LR", "RF", "LR_cal", "RF_cal"])]
    if sub.empty:
        return
    pref = "leave_one_scenario_out" if (sub.protocol == "leave_one_scenario_out").any() else "task_grouped"
    best = sub[sub.protocol == pref].sort_values("pr_auc", ascending=False).iloc[0]["model"]
    lines = [f"# Error analysis (K={primary_k}, model={best}, FPR={FPR_TARGET:.0%})\n"]
    for protocol in sub["protocol"].unique():
        d = oof_all[(oof_all.K == primary_k) & (oof_all.protocol == protocol)].copy()
        if d.empty:
            continue
        thr = threshold_at_fpr(d["y"].to_numpy(), d[best].to_numpy())
        d["flag"] = d[best] > thr
        fn, fp = d[(d.y == 1) & ~d.flag], d[(d.y == 0) & d.flag]
        lines.append(f"\n## {protocol}\n\nthreshold={thr:.4f}; missed risky agent-runs (FN)={len(fn)}; "
                     f"false alarms (FP)={len(fp)}.\n")
        if (d.y == 1).any():
            lines.append("Recall by attack type:\n\n" + d[d.y == 1].groupby("attack_type")["flag"].agg(
                n_pos="size", recall="mean").round(3).to_markdown() + "\n")
            lines.append("Recall by scenario:\n\n" + d[d.y == 1].groupby("scenario")["flag"].agg(
                n_pos="size", recall="mean").round(3).to_markdown() + "\n")
        lines.append("False alarms by scenario:\n\n" + fp.groupby("scenario").size().rename("FP").to_markdown() + "\n")
        lines.append("Ten lowest-scoring missed risky agent-runs:\n\n" +
                     fn.sort_values(best).head(10)[["run_id", "agent", "scenario", "attack_type", "first_attack_idx", best]]
                     .round(4).to_markdown(index=False) + "\n")
    a_ = res[(res.K == primary_k) & (res.model == best)].set_index("protocol")["pr_auc"]
    lines.append(f"\n## Generalisation gap\n\nPR-AUC task-grouped={a_.get('task_grouped', float('nan'))} vs "
                 f"leave-one-scenario-out={a_.get('leave_one_scenario_out', float('nan'))}. "
                 "A large drop means the model learned scenario-specific habits, not attack behaviour.\n")
    open(f"{out}/error_analysis.md", "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
