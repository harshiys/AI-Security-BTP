# Cyber-1 Label Review — defensibility of ML-A's Checkpoint 2 labels

**Role:** Cyber-1 reviews ML-A's labels for defensibility (work-division job #4):
*does the target `risky` really reflect misbehavior?*

**Scope note:** numbers below are from the **synthetic** sample output
(`simulate_logs.py`, no LLM). They validate the label *logic*, not real results.
Re-run this review on real logs before trusting any number.

---

## The labels ML-A ships

| Label | Meaning | Role |
|---|---|---|
| `agent_invoked_attack_tool` | agent called a TAMAS attack-tool in an attack run | **main target (y)** |
| `first_attack_idx` | event index of that first attack call | lead-time / "already harmed" exclusion |
| `agent_compromised_cfg` | agent's description was corrupted in the task file | auxiliary (error analysis) |
| `run_has_attack`, `attack_type` | run-level | per-attack recall |

The main target is **`agent_invoked_attack_tool`** — a behavioral fact ("did this
agent actually call a malicious tool"), which is the defensible choice: it is
grounded in an observed action, not an assumption.

---

## Findings

### Finding 1 — Target is defensible, but it only covers tool-based attacks
`agent_invoked_attack_tool` is sound for **Byzantine / Colluding / Contradicting**,
where misbehavior shows up as a malicious tool call. But **DPI / IPI /
Impersonation invoke no attack tool** (confirmed: those three contribute 0 to
`agent_compromised_cfg`, and their attacks live in the query/tool-output/text).
So for half the attack types, the main target is **structurally 0** — the model
cannot learn them from this label. This is not a bug, but it must be stated:
*the behavioral target detects behavioral attacks only.*

### Finding 2 — `agent_compromised_cfg` ≠ `agent_invoked_attack_tool` (expected, but watch it)
Crosstab on the sample:

| | invoked_tool=0 | invoked_tool=1 |
|---|---|---|
| compromised_cfg=False | 1323 | 71 |
| compromised_cfg=True | 187 | 63 |

Two things to note:
- **187 "compromised" agents never invoked an attack tool.** Defensible: an agent
  can be *configured* malicious yet not get its turn to act within the window.
  Correct to keep these out of the main target (they didn't misbehave *yet*).
- **71 "non-compromised" agents DID invoke an attack tool.** This needs a reason.
  On synthetic data it is the generator assigning tools loosely. On **real** data,
  if this persists, it signals either (a) label noise in `agent_compromised_cfg`
  recovery, or (b) genuinely a clean agent tricked into a bad call (DPI/IPI
  effect). Cyber-1 flags this cell for re-check on real logs.

### Finding 3 — `first_attack_idx` and the "already-harmed" exclusion is correct
Only rows with an attack call have a `first_attack_idx`; it is used to exclude
agent-runs whose harm already happened before the decision point K. This is the
right call — counting a detection *after* harm would inflate recall. Defensible.

### Finding 4 — No leakage from label into features (verified)
Confirmed with ML-A's design and Cyber-1's own `selfcheck.py`: features never
read tool names, and labels are joined only after feature construction. The
target is built from attack-tool names read separately. Good.

---

## Recommendations to ML-A

1. **State the target's scope explicitly** in the eval report: `agent_invoked_
   attack_tool` measures behavioral (tool-based) misbehavior; DPI/IPI/
   Impersonation are expected near-zero recall on this target and must be
   evaluated by Checkpoint 3 / Cyber-2, not counted against Checkpoint 2.
2. **On real logs, audit the 71-cell** (non-compromised agents invoking attack
   tools). If it stays large, investigate `agent_compromised_cfg` recovery.
3. **Report per-attack recall split into two groups** — behavioral attacks
   (Byzantine/Colluding/Contradicting) vs text attacks (DPI/IPI/Impersonation) —
   so the headline number is not dragged down by attacks this checkpoint
   structurally cannot catch.
4. Keep `agent_compromised_cfg` as auxiliary only (as ML-A already does); it is
   too noisy to be a training target.

**Verdict:** the main target is defensible and well-constructed. The one real
risk is presenting a single pooled recall across all six attacks, which would
unfairly punish Checkpoint 2 for text attacks it was never meant to catch
behaviorally. Split the reporting.
