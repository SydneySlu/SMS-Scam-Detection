# predict_with_scratch.py
import os
import joblib
import numpy as np
from model_from_scratch import LogisticRegressionScratch

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载向量器
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

# 加载 scratch 模型的参数
params = joblib.load(os.path.join(BASE_DIR, "models", "scratch_model.pkl"))

# 重新实例化并加载参数
scratch_model = LogisticRegressionScratch()
scratch_model.weights = params["weights"]
scratch_model.bias = params["bias"]

def predict_messages(messages):
    """
    输入一组短信文本，输出预测结果
    """
    # 文本转向量
    X = vectorizer.transform(messages)

    # 预测概率
    probs = scratch_model.predict_proba(X)

    # 概率 >= 0.5 判为 spam
    preds = (probs >= 0.5).astype(int)

    results = []
    for msg, pred, prob in zip(messages, preds, probs):
        label = "Spam" if pred == 1 else "Ham"
        results.append((msg, label, prob))
    return results

if __name__ == "__main__":
    test_messages = [
        "Congratulations! You have won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now.",
        "Hey, are we still meeting for lunch today?",
        "Free entry in 2 a weekly competition to win FA Cup final tickets. Text to enter now!"
    ]
    predictions = predict_messages(test_messages)
    for msg, label, prob in predictions:
        print(f"Message: {msg}\nPrediction: {label} (prob={prob:.4f})\n")