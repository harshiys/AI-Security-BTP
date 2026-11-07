# What changed in the attached CP2 package, and why

Architecture, feature code, graph code, trust-layer flow and scorer were kept. Changes are the minimum needed for real TAMAS data and the shared schema.

| File | Change | Why |
|---|---|---|
| `schema/shared_schema_format.json` | **added** (team's file, unchanged); `schema/risk_record.schema.json` **removed** | one contract only; my earlier schema was a competing one with `allow` instead of `pass` |
| `cp2_contract.py` | **new** | validates every SentinelMesh-facing record against the shared schema (5 fields, `pass/flag/block`, UTC timestamp) |
| `trust_layer.py` | verdict log now schema-only (no extra `run_id`, `allow`->`pass`, UTC timestamps); alerts renamed `cp2_alerts_internal.jsonl`; primary model LR_cal; data-source banner; mapping in `cp2_item_index_internal.csv` | schema rules forbid extra fields; LR first as requested |
| `scorer.py` | uses the contract; `allow`->`pass` | same |
| `mla_pipeline.py` | `--data_source`, `--label_mode`; marker-based detection and refusal to mix; new `verified_dpi` label (DPI only, per agent, from TAMAS's own tool-name regex and log rule); class-balance gate; accuracy/precision/recall/F1/ROC-AUC table; banners; clean message when no logs | the old label used a tool list containing 123 benign tools and applied the attack type to every agent; synthetic and real paths could mix |
| `ml_b_eval.py` | LR_cal primary (RF_cal kept as comparison); full classification metrics at the validation-chosen threshold; banners; gate; note about harmless-run features | requirements 4-6 |
| `metrics_extra.py` | **new** | small, shared metric helper |
| `tamas_inspect.py` | **new** | stage 2: what TAMAS gives, what labels mean, what is broken |
| `tamas_run_subset.py` | **new** | stage 3: small configurable real run; never fabricates; documents blockers |
| `tests/` | **new** (14 tests, `python -m unittest discover -s tests -t .`) | parsing incl. genuine AutoGen format, labels, leakage, contract, guards, blocked runner |
| `reports/` | **new** | inspection, blocked-execution status, results report |
| `README_CP2_MLA_MLB.md` | rewritten; adds **TAMAS Ground-Truth Labels and Limitations** | requirement 3 |
| `simulate_logs.py`, `graph_features.py`, `cybersec_adapters.py` | unchanged | working code reused |
| CP1 / WeightWatch | untouched | out of scope |

---

## Groq real-data workflow repair (this update)

Minimal changes to make the real-TAMAS runner work with Groq's OpenAI-compatible
API on Windows PowerShell. Architecture, schema, ML-A/ML-B, trust layer and the
data-integrity guards are unchanged.

| File | Change | Why |
|---|---|---|
| `patches/autogen_run.py` | **new** — drop-in replacement for `TAMAS/autogen_run.py`: builds a Groq `OpenAIChatCompletionClient` (base_url `https://api.groq.com/openai/v1`, key `GROQ_API_KEY`, `model_info` family `gpt-oss`); wires the correct tool module to every agent in **all five** DPI scenarios (was News-only); exposes only functions actually defined in each module (via `inspect`, no invented tools); handles ` A`/` B` agent suffixes; clear error if `GROQ_API_KEY` is missing | the stock runner used `OPENAI_API_KEY`, built an Azure client at import, and attached tools only to News agents, so non-News scenarios produced no tool calls (hence no labels) |
| `tamas_run_subset.py` | repaired: defaults to Groq model `openai/gpt-oss-20b`, validates `GROQ_API_KEY` (no longer demands `OPENAI_API_KEY`); added `--overwrite` (off by default, won't clobber existing logs), `--timeout`, richer `real_execution_status.json` (provider, model, per-run exit code, log path, byte count, overall status); useful errors for missing task JSON / key / autogen / API / timeout | requirement 3; never fabricates a log |
| `tests/test_groq_runner.py` | **new** — 5 tests: dry-run writes nothing, preflight names GROQ (never OPENAI), blocked path writes status + exits 2, nested log path, `--overwrite` accepted. No real API calls | requirement 8 |
| `README_CP2_MLA_MLB.md` | added **Windows PowerShell — real Groq workflow** section (install, set key, smoke test, 10 runs × 5 scenarios, check logs, ML-A, then ML-B/trust only if training completed, tests, common failures) + short Git Bash equivalents | requirement 4 |

### Known limitations (unchanged by this update)
- `verified_dpi` means **an agent requested the tool named in the injected DPI
  instruction** — not that the attack succeeded or that the agent is malicious.
- TAMAS ships no execution logs, no labels, and no harmless tasks; run-level
  negatives are missing, so the class-balance gate (>=10 positive / >=10
  negative / >=10 independent tasks) often will not be met from a small subset.
  **The gate is not lowered.** Collect more independent real tasks/configs.
- ML-A/ML-B are downstream scikit-learn classifiers/feature models. This is
  **not** fine-tuning Groq's hosted LLM.
- IPI has no TAMAS eval script; DPI is the only attack with a defensible
  per-agent label from the repo.
