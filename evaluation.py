"""
evaluation.py
功能：通用的模型评估与可视化（支持二分类/多分类）
"""

from typing import Any, Dict, Optional
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)


def evaluate_model(
    y_true,
    y_pred,
    model_name: str = "Model",
    average: str = "binary",
    digits: int = 4,
    target_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    打印并返回模型的评估指标（支持二分类/多分类）
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        average: 二分类用 "binary"，多分类可用 "macro" 或 "weighted"
        digits: 小数位数
        target_names: 标签名称
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    print(f"[{model_name}] accuracy={accuracy:.{digits}f} precision={precision:.{digits}f} recall={recall:.{digits}f} f1={f1:.{digits}f}")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=digits))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[list] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    绘制并保存混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(np.unique(y_true))))
    if labels is None:
        labels = [str(l) for l in np.unique(y_true)]

    plt.figure(figsize=(4.5, 4))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                     xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def _get_scores_for_roc(model: Any, X_test) -> np.ndarray:
    """优先使用 predict_proba 的正类概率；否则使用 decision_function。"""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_test))
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]  # 二分类取正类
        return proba.ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test)).ravel()
    return np.asarray(model.predict(X_test)).ravel()


def plot_roc_curve(
    model: Any,
    X_test,
    y_test,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    绘制并保存 ROC 曲线（二分类）
    """
    scores = _get_scores_for_roc(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, ls="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

    return {"auc": float(roc_auc)}

