"""
model_sklearn.py
功能：使用 sklearn Logistic Regression 训练和预测
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import scipy.sparse as sp


def train_sklearn_model(X_train, y_train, **kwargs):
    """
    使用 sklearn 的逻辑回归模型训练
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签
        **kwargs: 传递给 LogisticRegression 的额外参数
    返回:
        model: 训练好的模型
    """
    # 设置默认参数，支持稀疏矩阵
    default_params = {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'liblinear',  # 适合小数据集和稀疏矩阵
        'class_weight': 'balanced'  # 处理类别不平衡
    }
    
    # 合并用户参数和默认参数
    params = {**default_params, **kwargs}
    
    # 创建并训练模型
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    return model


def predict_sklearn_model(model, X_test):
    """
    输入:
        model: 已训练的 sklearn 模型
        X_test: 测试集特征
    输出:
        y_pred: 预测结果
    """
    if sp.issparse(X_test):
        X_test = X_test.tocsr()
    return model.predict(X_test)


def predict_proba_sklearn_model(model, X_test):
    """
    输入:
        model: 已训练的 sklearn 模型
        X_test: 测试集特征
    输出:
        y_proba: 预测概率
    """
    if sp.issparse(X_test):
        X_test = X_test.tocsr()
    try:
        return model.predict_proba(X_test)
    except AttributeError:
        # 某些 solver 可能不支持 predict_proba
        return model.predict_proba(X_test)


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    参数:
        model: 已训练的模型
        X_test: 测试特征
        y_test: 真实标签
    返回:
        dict: 包含准确率、分类报告、混淆矩阵的字典
    """
    X_test = X_test.tocsr()

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        y_proba = model.decision_function(X_test)

    # 动态生成标签名字
    labels = unique_labels(y_test, y_pred)
    target_names = [str(lbl) for lbl in labels]

    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    return {
        'accuracy': accuracy,
        'classification_report_str': report_str,
        'classification_report_dict': report_dict,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba
    }
