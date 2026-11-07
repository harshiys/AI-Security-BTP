"""Plain classification metrics at a fixed threshold (CP2). Small on purpose: sklearn does the work."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def class_balance(y):
    y = np.asarray(y).astype(int)
    return dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                prevalence=round(float(y.mean()), 4) if len(y) else float("nan"))


def classification_metrics(y, score, thr):
    """accuracy / precision / recall / F1 at `thr`; ROC-AUC and PR-AUC from the raw scores.
    A metric that cannot be computed (one class only) is NaN, never 0."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    pred = (s > thr).astype(int)
    out = class_balance(y)
    both = out["n_pos"] > 0 and out["n_neg"] > 0
    out.update(
        accuracy=round(accuracy_score(y, pred), 4),
        precision=round(precision_score(y, pred, zero_division=0), 4) if out["n_pos"] else float("nan"),
        recall=round(recall_score(y, pred, zero_division=0), 4) if out["n_pos"] else float("nan"),
        f1=round(f1_score(y, pred, zero_division=0), 4) if out["n_pos"] else float("nan"),
        roc_auc=round(roc_auc_score(y, s), 4) if both else float("nan"),
        pr_auc=round(average_precision_score(y, s), 4) if out["n_pos"] else float("nan"))
    return out
