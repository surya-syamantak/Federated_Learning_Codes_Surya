"""Classification metrics from a binary confusion matrix (1 = heart disease).

Every metric is computed from the four counts TP, FP, TN, FN, so clients only
need to report those counts and the server can also pool them across clients.
A metric whose denominator is 0 is reported as 0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "mcc", "kappa")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    n = tp + fp + tn + fn
    accuracy = _div(tp + tn, n)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)  # sensitivity
    specificity = _div(tn, tn + fp)
    f1 = _div(2 * precision * recall, precision + recall)
    mcc = _div(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    # Cohen's kappa: observed agreement vs agreement expected by chance.
    p_chance = _div((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), n * n)
    kappa = _div(accuracy - p_chance, 1 - p_chance)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1, "mcc": mcc, "kappa": kappa}


def metrics_table(per_client: dict[str, dict]) -> pd.DataFrame:
    """One row per client, plus pooled ("All clients") and macro-average rows.

    ``per_client`` maps client name to a dict holding tp/fp/tn/fn and loss.
    Pooled metrics come from the summed confusion matrix; the macro average
    is the unweighted mean of the client rows.
    """
    rows = []
    for name, m in per_client.items():
        counts = {k: m[k] for k in ("tp", "fp", "tn", "fn")}
        rows.append({"client": name, "n_val": sum(counts.values()), **counts,
                     "log_loss": m["loss"], **metrics_from_counts(**counts)})
    table = pd.DataFrame(rows)
    pooled_counts = {k: int(table[k].sum()) for k in ("tp", "fp", "tn", "fn")}
    pooled_loss = float((table["log_loss"] * table["n_val"]).sum() / table["n_val"].sum())
    pooled = {"client": "All clients (pooled)", "n_val": sum(pooled_counts.values()),
              **pooled_counts, "log_loss": pooled_loss, **metrics_from_counts(**pooled_counts)}
    macro = {"client": "Macro average", "log_loss": table["log_loss"].mean(),
             **{k: table[k].mean() for k in METRICS}}
    return pd.concat([table, pd.DataFrame([pooled, macro])], ignore_index=True)


def format_table(table: pd.DataFrame) -> str:
    cols = ["client", "n_val", "log_loss", *METRICS]
    shown = table[cols].copy()
    shown["n_val"] = shown["n_val"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return shown.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def summarize_runs(tables: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the final_metrics tables of repeated runs (one per seed).

    Returns (all_runs, summary): every run's rows with a ``seed`` column, and
    per client the mean and sample standard deviation of each metric.
    """
    all_runs = pd.concat([t.assign(seed=s) for s, t in tables.items()], ignore_index=True)
    cols = ["log_loss", *METRICS]
    order = list(dict.fromkeys(all_runs["client"]))  # keep client order
    grouped = all_runs.groupby("client", sort=False)[cols]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    summary.insert(0, "n_runs", all_runs.groupby("client", sort=False).size())
    summary = summary.loc[order].reset_index()
    return all_runs, summary[["client", "n_runs", *[f"{c}_{s}" for c in cols for s in ("mean", "std")]]]


def format_summary(summary: pd.DataFrame) -> str:
    """Render the summary as ``mean ± std`` per metric."""
    shown = summary[["client"]].copy()
    for c in ["log_loss", *METRICS]:
        shown[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_std"])]
    return shown.to_string(index=False)
