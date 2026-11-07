"""Scoring, validation-based thresholds, metrics, and explanations."""
from __future__ import annotations

import numpy as np


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def raw_scores(bundle: dict, X: np.ndarray) -> np.ndarray:
    return bundle["estimator"].predict_proba(X)[:, 1]


def final_scores(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Risk score in [0,1] (1.0 = highest risk): raw model probability, Platt-calibrated on validation."""
    p = raw_scores(bundle, X)
    platt = bundle.get("platt")
    if platt is not None:
        p = platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    return np.clip(p, 0.0, 1.0)


def choose_thresholds(y, s, fpr_block: float = 0.01, recall_flag: float = 0.95) -> dict:
    """Pick thresholds from VALIDATION scores only.

    t_block: lowest threshold whose validation FPR <= fpr_block   (BLOCK = high-precision hard stop)
    t_flag : highest threshold whose validation recall >= recall_flag (FLAG+BLOCK = high-recall net)
    Thresholds sit midway between neighbouring observed scores to leave a margin.
    """
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError("validation set needs both classes to choose thresholds")
    cands = np.unique(np.concatenate([s, [0.0, 1.0]]))
    fpr = np.array([(neg >= t).mean() for t in cands])
    rec = np.array([(pos >= t).mean() for t in cands])
    i = np.where(fpr <= fpr_block)[0]
    ib = int(i[0]) if len(i) else len(cands) - 1
    t_block = float((cands[ib - 1] + cands[ib]) / 2) if ib > 0 else float(cands[ib])
    j = np.where(rec >= recall_flag)[0]
    jf = int(j[-1]) if len(j) else 0
    t_flag = float((cands[jf - 1] + cands[jf]) / 2) if jf > 0 else float(cands[jf])
    t_flag = min(t_flag, t_block)
    return {
        "t_flag": t_flag, "t_block": t_block,
        "fpr_block_target": fpr_block, "recall_flag_target": recall_flag,
        "val_fpr_at_block": float((neg >= t_block).mean()), "val_recall_at_block": float((pos >= t_block).mean()),
        "val_fpr_at_flag": float((neg >= t_flag).mean()), "val_recall_at_flag": float((pos >= t_flag).mean()),
        "val_n_benign": int(len(neg)), "val_n_backdoor": int(len(pos)),
        "chosen_on": "validation split only",
    }


def verdict_from_score(score: float, th: dict) -> str:
    if score >= th["t_block"]:
        return "block"
    if score >= th["t_flag"]:
        return "flag"
    return "pass"


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion k/n (honest uncertainty for small test sets)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def binary_metrics(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp and tp + fn and (prec + rec) > 0) else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "recall_ci95": wilson(tp, tp + fn), "fpr_ci95": wilson(fp, fp + tn)}


class Explainer:
    """Local explanation for one adapter's feature vector. Positive contribution = pushes toward 'backdoor'."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.cols = bundle["feature_cols"]
        self.shap_explainer = None
        if bundle["model_name"] == "rf":
            try:
                import shap  # optional
                self.shap_explainer = shap.TreeExplainer(bundle["estimator"])
            except Exception:  # noqa: BLE001
                self.shap_explainer = None

    def contributions(self, x: np.ndarray) -> tuple[np.ndarray, str]:
        b = self.b
        if b["model_name"] == "lr":
            sc, clf = b["estimator"].named_steps["scaler"], b["estimator"].named_steps["clf"]
            return clf.coef_[0] * ((x - sc.mean_) / sc.scale_), "logreg_coef_x_standardized_value"
        if self.shap_explainer is not None:
            sv = self.shap_explainer.shap_values(x.reshape(1, -1))
            if isinstance(sv, list):
                return np.asarray(sv[1])[0], "shap_tree"
            sv = np.asarray(sv)
            return (sv[0, :, 1] if sv.ndim == 3 else sv[0]), "shap_tree"
        mu, sd = b["benign_mean"], b["benign_std"]
        z = np.clip((x - mu) / np.where(sd > 0, sd, 1.0), -10, 10)
        return b["estimator"].feature_importances_ * z, "rf_importance_x_benign_zscore (install shap for exact values)"

    def top(self, x: np.ndarray, k: int = 5) -> tuple[list[dict], str]:
        c, method = self.contributions(x)
        order = np.argsort(-c)[:k]
        return [{"feature": self.cols[i], "value": float(x[i]), "contribution": float(c[i])} for i in order], method
