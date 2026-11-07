> **Data source: real TAMAS execution logs. Label definition: verified_dpi** (agent requested the tool that the injected DPI instruction names; see README section 'TAMAS Ground-Truth Labels and Limitations'). This is not 'attack success' and not 'agent is malicious'.

# Evaluation (decision point K=1 events, recall at FPR=5%)

| protocol     | model   |   already_harmed_excluded |   n |   n_pos |   prevalence |   pr_auc |   recall_at_fpr |   achieved_fpr |    brier |
|:-------------|:--------|--------------------------:|----:|--------:|-------------:|---------:|----------------:|---------------:|---------:|
| task_grouped | Rule    |                         0 |  40 |       3 |        0.075 |   0.0667 |               0 |              0 | nan      |
| task_grouped | LR      |                         0 |  40 |       3 |        0.075 |   0.0736 |               0 |              0 |   0.3225 |
| task_grouped | RF      |                         0 |  40 |       3 |        0.075 |   0.0539 |               0 |              0 |   0.1449 |
| task_grouped | LR_cal  |                         0 |  40 |       3 |        0.075 |   0.0736 |               0 |              0 |   0.3225 |
| task_grouped | RF_cal  |                         0 |  40 |       3 |        0.075 |   0.0539 |               0 |              0 |   0.1449 |

## Early detection: recall at fixed FPR as the decision point moves later

| protocol     | model   |   K |   n_pos |   recall_at_fpr |   pr_auc |   already_harmed_excluded |
|:-------------|:--------|----:|--------:|----------------:|---------:|--------------------------:|
| task_grouped | LR_cal  |   1 |       3 |               0 |   0.0736 |                         0 |
| task_grouped | RF_cal  |   1 |       3 |               0 |   0.0539 |                         0 |
| task_grouped | LR_cal  |   2 |       1 |               0 |   0.0263 |                         2 |
| task_grouped | RF_cal  |   2 |       1 |               0 |   0.0263 |                         2 |
| task_grouped | LR_cal  |   4 |       1 |               0 |   0.0263 |                         2 |
| task_grouped | RF_cal  |   4 |       1 |               0 |   0.0263 |                         2 |
| task_grouped | LR_cal  |   8 |       0 |             nan | nan      |                         3 |
| task_grouped | RF_cal  |   8 |       0 |             nan | nan      |                         3 |

## Per-attack recall (K=1, calibrated models)

| protocol     |   K | model   | attack_type   |   n_pos |   recall |
|:-------------|----:|:--------|:--------------|--------:|---------:|
| task_grouped |   1 | LR_cal  | DPI           |       3 |        0 |
| task_grouped |   1 | RF_cal  | DPI           |       3 |        0 |

## Accuracy / precision / recall / F1 / ROC-AUC (LR first, RF for comparison)

| protocol     | model   |   K |   threshold |   n |   n_pos |   n_neg |   prevalence |   accuracy |   precision |   recall |   f1 |   roc_auc |   pr_auc |
|:-------------|:--------|----:|------------:|----:|--------:|--------:|-------------:|-----------:|------------:|---------:|-----:|----------:|---------:|
| task_grouped | LR_cal  |   1 |      0.8118 |  40 |       3 |      37 |        0.075 |      0.925 |           0 |        0 |    0 |    0.3739 |   0.0736 |
| task_grouped | RF_cal  |   1 |      0.5084 |  40 |       3 |      37 |        0.075 |      0.925 |           0 |        0 |    0 |    0.1216 |   0.0539 |

Class balance: {'n': 40, 'n_pos': 3, 'n_neg': 37, 'prevalence': 0.075, 'n_tasks': 10, 'n_runs': 10, 'data_source': 'tamas_real', 'label_mode': 'verified_dpi'}. Accuracy alone is misleading when positives are rare.
