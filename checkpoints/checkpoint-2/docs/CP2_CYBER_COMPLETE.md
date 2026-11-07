# SentinelMesh Checkpoint 2 — Complete Cyber Layer

## Scope
This package implements the Cyber-side CP2 components represented in the Module 2 diagram:

1. Behavioral Fingerprint — implemented on the supplied real TAMAS-derived logs.
2. Communication Graph — extracts the observed agent-to-agent graph and graph telemetry.
3. Message Provenance — implements the runtime cryptographic protocol and explicitly audits the historical logs as `unavailable` because they contain no signatures.
4. Target attack scenarios — controlled evaluation traces for impersonation, rogue delegation and sleeper-agent behavior.
5. Cyber-side fusion adapter — converts the three signals into an interpretable cyber risk signal for the ML-B / trust layer.

## Important data boundary
The historical real TAMAS-derived logs contain 10 DPI runs and 40 agent-run label records. They are useful for real behavioral/graph telemetry, but they do not contain cryptographic signatures and do not provide verified benign-agent ground truth. Therefore this package does **not** fabricate provenance evidence or claim that a historical run was successfully attacked merely because the task was an attack instance.

The controlled attack scenarios are generated from the real TAMAS roster and are clearly marked as evaluation scenarios.

## Architecture
`events.csv -> communication_graph.py -> graph telemetry`

`events.csv -> existing real Cyber-1 fingerprint -> behavioral drift`

`runtime message -> provenance.py -> signature verification`

`behavioral + graph + provenance -> cyber_fusion.py -> cyber risk -> ML-B / Dynamic Agent Trust Score`

## Run
```bash
python run_cyber_cp2.py
```

No external API or LLM is required for this Cyber-side processing.

## What remains outside this package
The ML-B/GNN training model and the final project-wide Global Risk-Fusion implementation belong to the ML/integration side. This package supplies the Cyber telemetry and integration signal they consume.
