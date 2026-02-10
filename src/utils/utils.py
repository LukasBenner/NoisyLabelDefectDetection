import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    
    if not metric_name:
        log.info("No optimized metric specified; skipping metric retrieval.")
        return None

    if metric_name not in metric_dict:
        raise KeyError(f"Metric '{metric_name}' not found in the metric dictionary.")
    
    return metric_dict[metric_name]

def calculate_summary_statistics(
    all_metrics: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate summary statistics across multiple runs."""
    all_metrics_df = pd.DataFrame(all_metrics)
    summary_statistics = []
    metrics_to_analyze = [m for m in all_metrics_df.columns if m not in ["run_idx"]]

    for metric in metrics_to_analyze:
        values = all_metrics_df[metric].dropna()
        n = len(values)
        mean = values.mean()
        median = values.median()
        std = values.std(ddof=1)
        se = std / np.sqrt(n) if n > 0 else 0.0

        if n > 1:
            t = stats.t.ppf(0.975, df=n - 1)
            margin = t * se
            ci_lower = mean - margin
            ci_upper = mean + margin
        else:
            ci_lower = ci_upper = mean

        summary_statistics.append(
            {
                "metric": metric,
                "mean": mean,
                "median": median,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "min": values.min(),
                "max": values.max(),
            }
        )

    return all_metrics_df, pd.DataFrame(summary_statistics)

def to_float(value):
        """Convert tensor or numeric value to float."""
        if value is None:
            return None
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

def create_confusion_matrix(
    preds,
    targets,
    class_names,
    logging_directory,
    max_classes_for_plot: int = 200,
    annotate_threshold: int = 50,
):
    """Create and save confusion matrix assets.

    For many classes, rendering a fully annotated heatmap is slow and hard to read.
    This helper skips or de-annotates plots when class count is large.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(logging_directory, exist_ok=True)
    num_classes = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    if num_classes > max_classes_for_plot:
        log.warning(
            "Skipping confusion matrix plots (%d classes > %d). Saving raw arrays instead.",
            num_classes,
            max_classes_for_plot,
        )
        np.save(os.path.join(logging_directory, "confusion_matrix.npy"), cm)
        cm_normalized = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        np.save(
            os.path.join(logging_directory, "confusion_matrix_normalized.npy"),
            cm_normalized,
        )
        return

    annotate = num_classes <= annotate_threshold
    fmt = "d" if annotate else ""
    fig_size = max(8, min(2 * num_classes, 40))

    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        cm,
        annot=annotate,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(logging_directory, "confusion_matrix.png"))
    plt.close()

    cm_normalized = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        cm_normalized,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(logging_directory, "confusion_matrix_normalized.png"))
    plt.close()