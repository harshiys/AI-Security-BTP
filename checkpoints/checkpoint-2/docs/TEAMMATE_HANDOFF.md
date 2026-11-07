# Cyber -> ML-B handoff

Use these files:

- `results/cyber1_fingerprint_drift.jsonl`: existing real-data behavioral drift, keyed by `(run_id, agent)`.
- `results/communication_graph.csv`: communication-graph features, keyed by `(run_id, agent)`.
- `results/provenance_real.csv`: historical provenance audit. `source=unavailable` means the source logs did not record cryptographic signatures; do not interpret this as an unsigned attack.
- `results/cyber_cp2_fusion.csv`: optional Cyber-side interpretable fusion signal.
- `results/cp2_attack_scenarios.json`: controlled attack cases for testing impersonation, rogue delegation and sleeper-agent handling.

Recommended integration: merge the first two tables on `(run_id, agent)` and use the Cyber-side values as additional evidence alongside the ML-B score. Do not overwrite ML-B's learned risk with `cyber_risk`.
