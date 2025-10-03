"""
data_preprocessing.py
功能：加载短信诈骗数据集，清理文本，向量化，划分训练/测试集
"""

import re
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def _clean_text(text: str) -> str:
    """将文本小写化，并移除非字母字符，仅保留空格分隔的英文单词。"""
    if not isinstance(text, str):
        return ""
    # lowered = text.lower()
    # # 将非字母字符替换为空格，然后压缩多余空格
    # alnum_only = re.sub(r"[^a-z0-9]", " ", lowered)
    # compact = re.sub(r"\s+", " ", alnum_only).strip()

    text = text.lower()

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "http", text)
    text = re.sub(r"\S+@\S+", "email", text)
    text = re.sub(r"\+?\d[\d -]{7,}\d", lambda m: re.sub(r"\D", "", m.group()), text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()


    return text


def load_and_preprocess_data(
    filepath: str,
    balance_classes: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    加载并预处理短信数据
    参数:
        filepath (str): 数据文件路径 (.csv 或 .xlsx)
        balance_classes (bool): 是否在训练集上做类别平衡 (SMOTE)
        test_size (float): 测试集比例
        random_state (int): 随机种子
    返回:
        X_train, X_test, y_train, y_test, vectorizer
    """

    # 1. 读取 CSV 或 Excel
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, encoding="latin-1")
    elif filepath.endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")

    # 2. 兼容不同数据列名
    if "label" in df.columns and "message" in df.columns:
        labels = df["label"]
        texts = df["message"]
    elif "v1" in df.columns and "v2" in df.columns:
        labels = df["v1"]
        texts = df["v2"]
    else:
        raise ValueError("CSV must contain 'label/message' or 'v1/v2' columns")

    # 3. 标签转二进制
    label_mapping = {"ham": 0, "spam": 1}
    y = labels.astype(str).str.lower().map(label_mapping)
    if y.isna().any():
        raise ValueError("Found unknown labels; expected only 'ham' or 'spam'")

    # 4. 文本清洗
    texts = texts.fillna("").astype(str).map(_clean_text)

    # 5. 划分训练/测试集 (避免数据泄漏)
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. TF-IDF 向量化
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    # 7. 可选：类别平衡 (仅训练集上做)
    if balance_classes:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), vectorizer




