# ML-B: ablation, thresholds, hard rules

> **Data source: real TAMAS execution logs. Label definition: verified_dpi** (agent requested the tool that the injected DPI instruction names; see README section 'TAMAS Ground-Truth Labels and Limitations'). This is not 'attack success' and not 'agent is malicious'.

## Ablation by configuration (K=1; threshold chosen on validation tasks at 5% benign FPR)

| config     |   K | features   | model   |   n |   n_pos |   accuracy |   precision |   recall |   f1 |   roc_auc |   pr_auc |   fpr_at_val_thr |
|:-----------|----:|:-----------|:--------|----:|--------:|-----------:|------------:|---------:|-----:|----------:|---------:|-----------------:|
| RoundRobin |   1 | tab        | LR_cal  |  40 |       3 |      0.625 |           0 |        0 |    0 |    0.3018 |   0.0655 |           0.3243 |
| RoundRobin |   1 | tab+graph  | LR_cal  |  40 |       3 |      0.825 |           0 |        0 |    0 |    0.8468 |   0.25   |           0.1081 |
| RoundRobin |   1 | tab        | RF_cal  |  40 |       3 |      0.925 |           0 |        0 |    0 |    0.1937 |   0.0639 |           0      |
| RoundRobin |   1 | tab+graph  | RF_cal  |  40 |       3 |      0.9   |           0 |        0 |    0 |    0.8108 |   0.2177 |           0.027  |

Round Robin's fixed order may add little, so this is tested, not assumed. Only the configurations present in the logs appear.

## Does adding graph features help? paired bootstrap over tasks (LR_cal)

|   K | model   |   mean_gain |   ci_low |   ci_high |   p_gain_positive |
|----:|:--------|------------:|---------:|----------:|------------------:|
|   1 | LR_cal  |      0.2182 |   0.0799 |    0.4271 |                 1 |
|   2 | LR_cal  |      0      |   0      |    0      |                 0 |
|   4 | LR_cal  |      0      |   0      |    0      |                 0 |
|   8 | LR_cal  |      0      |   0      |    0      |                 0 |

## Threshold report: threshold chosen on validation, then checked on unseen test tasks

|   K |   fold |   val_fold | features   | model   |   threshold |   val_fpr |   test_fpr |   test_recall |   n_val |   n_test |
|----:|-------:|-----------:|:-----------|:--------|------------:|----------:|-----------:|--------------:|--------:|---------:|
|   1 |      0 |          1 | tab        | LR_cal  |      0.7159 |         0 |     0      |           nan |       8 |        8 |
|   1 |      0 |          1 | tab+graph  | LR_cal  |      0.9336 |         0 |     0      |           nan |       8 |        8 |
|   1 |      1 |          2 | tab        | LR_cal  |      0.9196 |         0 |     0.5714 |             0 |       8 |        8 |
|   1 |      1 |          2 | tab+graph  | LR_cal  |      0.9794 |         0 |     0.1429 |             0 |       8 |        8 |
|   1 |      2 |          3 | tab        | LR_cal  |      0.5767 |         0 |     0.5    |           nan |       8 |        8 |
|   1 |      2 |          3 | tab+graph  | LR_cal  |      0.6957 |         0 |     0.25   |           nan |       8 |        8 |
|   1 |      3 |          4 | tab        | LR_cal  |      0.5545 |         0 |     0.5714 |             0 |       8 |        8 |
|   1 |      3 |          4 | tab+graph  | LR_cal  |      0.8322 |         0 |     0.1429 |             0 |       8 |        8 |
|   1 |      4 |          0 | tab        | LR_cal  |      0.6477 |         0 |     0      |             0 |       8 |        8 |
|   1 |      4 |          0 | tab+graph  | LR_cal  |      0.9144 |         0 |     0      |             0 |       8 |        8 |

## Hard rules (block): limit = largest benign value on validation tasks

| rule                |   fold |   limit |   test_fpr |   test_recall |   n_fired |
|:--------------------|-------:|--------:|-----------:|--------------:|----------:|
| g_pingpong > 0      |      0 |       0 |          0 |           nan |         0 |
| g_n_novel_edges > 0 |      0 |       0 |          0 |           nan |         0 |
| a_n_err > 0         |      0 |       0 |          0 |           nan |         0 |
| g_pingpong > 0      |      1 |       0 |          0 |             0 |         0 |
| g_n_novel_edges > 0 |      1 |       0 |          0 |             0 |         0 |
| a_n_err > 0         |      1 |       0 |          0 |             0 |         0 |
| g_pingpong > 0      |      2 |       0 |          0 |           nan |         0 |
| g_n_novel_edges > 0 |      2 |       0 |          0 |           nan |         0 |
| a_n_err > 0         |      2 |       0 |          0 |           nan |         0 |
| g_pingpong > 0      |      3 |       0 |          0 |             0 |         0 |
| g_n_novel_edges > 0 |      3 |       0 |          0 |             0 |         0 |
| a_n_err > 0         |      3 |       0 |          0 |             0 |         0 |
| g_pingpong > 0      |      4 |       0 |          0 |             0 |         0 |
| g_n_novel_edges > 0 |      4 |       0 |          0 |             0 |         0 |
| a_n_err > 0         |      4 |       0 |          0 |             0 |         0 |

## Final bundle
feature set: **tab+graph**, model LR_cal, K=1, flag threshold (risk) 0.9144 (median of per-fold validation thresholds), hard-rule limits {"g_pingpong": 0.0, "g_n_novel_edges": 0.0, "a_n_err": 0.0}.

GNN decision: **NO-GO** (see mlb_gnn_decision.md)
