# %% [markdown]
# # evaluate.py: the one way every model in the group is scored
#
# Import this from any training script (LSTM, CNN, Random Forest, SelfPAB):
#
#     from evaluate import score, print_report, plot_confusion
#     res = score(y_true, y_pred)          # dict of metrics
#     print_report(res, "HAR70+")
#
# `y_true` and `y_pred` are integer class indices into `config.CLASSES`.
#
# Why these metrics (from the project plan):
#  - macro-F1: the F1 of every class averaged with equal weight, so the two
#    tiny stairs classes count as much as walking. Accuracy would be
#    dominated by walking and sitting.
#  - per-class recall: "of all true stairs_up windows, how many did we
#    catch?" - shows which activities break for older adults.
#  - confusion matrix: where the errors go (e.g. stairs -> walking).

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

CLASSES = config.CLASSES
LABELS = list(range(len(CLASSES)))


def score(y_true, y_pred) -> dict:
    """All metrics in one dict. Classes absent from y_true get recall NaN."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    # A class that does not occur in y_true has no meaningful recall.
    recall = np.where(support > 0, recall, np.nan)
    present = support > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        # macro-F1 over the classes that actually occur in this test set
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.array(LABELS)[present],
                                   average="macro", zero_division=0)),
        "per_class_recall": dict(zip(CLASSES, recall.round(4).tolist())),
        "per_class_precision": dict(zip(CLASSES, precision.round(4).tolist())),
        "per_class_f1": dict(zip(CLASSES, f1.round(4).tolist())),
        "support": dict(zip(CLASSES, support.tolist())),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def print_report(res: dict, title: str = "") -> None:
    print(f"\n=== {title} ===  n = {res['n']:,}")
    print(f"macro-F1 {res['macro_f1']:.3f}   accuracy {res['accuracy']:.3f}")
    print(f"{'class':<12} {'recall':>7} {'precision':>10} {'f1':>6} {'support':>8}")
    for c in CLASSES:
        r = res["per_class_recall"][c]
        r_txt = "   -   " if r is None or np.isnan(r) else f"{r:7.3f}"
        print(f"{c:<12} {r_txt} {res['per_class_precision'][c]:10.3f} "
              f"{res['per_class_f1'][c]:6.3f} {res['support'][c]:8,}")


def plot_confusion(res: dict, title: str, path: Path, normalize: bool = True) -> None:
    """Confusion matrix as an image; rows = true class, columns = predicted."""
    cm = np.array(res["confusion"], dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row-wise %
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(LABELS); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(LABELS); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in LABELS:
        for j in LABELS:
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center",
                    fontsize=8, color="white" if v > (0.5 if normalize else cm.max() / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
