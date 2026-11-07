# CP2 results report: synthetic self-test vs real TAMAS

Generated from the run in this package. Four stages, reported separately.

## Stage status: what actually completed
| Stage | Completed? | Evidence |
|---|---|---|
| Unit tests (14) | **yes, all pass** | `python -m unittest discover -s tests -t .` |
| 2. TAMAS data inspection | **yes** | `reports/tamas_data_inspection.md` |
| 1. Synthetic pipeline self-test (ML-A, trust layer, ML-B, scorer, contract) | **yes** | `sample_output_SYNTHETIC/` (1200 fake logs, 4932 agent-runs) |
| 3. Actual TAMAS execution | **NO, blocked** | no LLM credentials; sandbox cannot reach api.openai.com. `reports/real_execution_status.md` |
| 4. Real TAMAS-derived training and held-out evaluation | **NO** | needs stage 3. **There are no real TAMAS results in this package.** |

## A. TAMAS data inspection (real repository facts)
- 300 attack instances (50 per attack, 5 scenarios). Each holds only `agents` and `user query`.
- No execution logs, no label or annotation file, no Harmless tasks, no IPI eval script are in the repo.
- Labels exist only as outcomes computed by `eval/*.py` from logs (run-level; needs an OpenAI judge).
- DPI: all 50 task texts name the injected tool; 47 of those tools exist in the tool files.
- The shipped runner attaches tools only to News agents; `autogen_inference.py` has a SyntaxError.
- Consequence: the supervised target is limited to DPI, news scenario, agent level (README, "TAMAS Ground-Truth Labels and Limitations").

## B. Synthetic self-test (NOT TAMAS results; invented labels, 3 configurations)
Purpose: show that the pipeline mechanics work. Fake logs come from `simulate_logs.py`; the "attack behaviour" in them is invented probabilities.
Positives are 8.0% of agent-runs at K=8 (390 of 4904); 338 independent tasks.

**ML-A, K=8, task-grouped folds, threshold at 5% benign FPR (pooled out-of-fold, slightly optimistic):**
| model | accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LR_cal (primary) | 0.882 | 0.145 | 0.097 | 0.116 | 0.590 | 0.110 |
| RF_cal (comparison) | 0.880 | 0.104 | 0.067 | 0.081 | 0.584 | 0.100 |

Reading it: accuracy is ~0.88 only because 92% of agent-runs are negative (always saying "negative" scores 0.92). ROC-AUC 0.59 is close to chance. On these fake logs the features barely separate the classes. High accuracy here would be meaningless anyway.

**Trust layer (LR_cal, benign-only threshold, 4 decision points):** flagged 231 agents; 39 of 418 risky agents (9.3%) and 192 of 4514 benign agents (4.3%). Verdict log: 4926 lines, **0 violate the shared schema** (checked with the team's file independently of my code), verdicts `pass` 4695 / `flag` 231.

**ML-B ablation (K=8, LR_cal, validation-chosen threshold, per configuration):** adding graph features changed PR-AUC by +0.0026 overall (95% task-bootstrap interval -0.0115 to +0.0141), i.e. **no clear gain**; per-configuration rows are in `mlb_report.md`. The simpler tabular feature set was therefore kept for the scorer bundle. Hard rules (ping-pong, novel edges, repeated errors) never fired on held-out tasks: they are untested on real patterns. **GNN decision: NO-GO** (gain not clear; 1200 runs < 2000).

## C. Real TAMAS evaluation
**None.** Blocked at stage 3. No label, metric or result in this package comes from real TAMAS executions.
