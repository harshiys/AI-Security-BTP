# GNN go / no-go note (ML-B with ML-A)

Rule agreed in advance: consider a GNN only if graph features CLEARLY help (PR-AUC gain >= 0.03 with the 95% task-bootstrap interval above 0) AND there is enough graph data (>= 2000 runs).

- PR-AUC gain from adding graph features (K=1, LR_cal): +0.2182 (95% CI +0.0799 to +0.4271; gain > 0 in 100% of resamples)
- runs available: 10 (needed >= 2000)
- configurations present in the data: 1

## Decision: **NO-GO**

Keep hand-built graph features plus the tree/linear models. Revisit when (a) real logs from several configurations (Swarm, CrewAI, orchestrator) exist and (b) the ablation shows a clear gain.
