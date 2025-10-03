"""
predict_friendly.py
用户友好预测脚本
功能：面向用户，提供多功能，多渠道的短信预测
支持：
  1. 命令行输入一条短信
  2. 直接在代码中写一个 list
  3. 从 txt 文件批量读取
"""

import os
import joblib
import sys

# ============ 加载模型和向量器 ============
MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "sklearn_model.pkl")  # 用 sklearn 训练好的模型

if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "缺少模型或向量器，请先运行 main.py 训练并保存模型！"
    )

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


# ============ 预测函数 ============
def predict_messages(messages):
    """
    输入: 短信列表
    输出: [(短信, 标签, 概率), ...]
    """
    X = vectorizer.transform(messages)
    probs = model.predict_proba(X)[:, 1]  # 取 spam 概率
    preds = (probs >= 0.5).astype(int)

    results = []
    for msg, pred, prob in zip(messages, preds, probs):
        label = "Spam" if pred == 1 else "Ham"
        results.append((msg, label, prob))
    return results


# ============ 入口 ============
if __name__ == "__main__":
    print("=== SMS Spam Detection ===")
    print("请选择模式：")
    print("1. 手动输入一句短信")
    print("2. 批量短信（代码里定义）")
    print("3. 从 txt 文件读取（一行一条短信）")

    choice = input("请输入选项 (1/2/3): ").strip()

    if choice == "1":
        msg = input("请输入短信内容: ").strip()
        results = predict_messages([msg])
    elif choice == "2":
        test_messages = [
            "Congratulations! You've won a free ticket. Reply WIN to claim now.",
            "Hey, are we still meeting tomorrow?",
            "URGENT! Your account has been compromised. Send us your details immediately.",
            "Good morning, just checking in. See you later!"
        ]
        results = predict_messages(test_messages)
    elif choice == "3":
        file_path = input("请输入 txt 文件路径: ").strip()
        if not os.path.exists(file_path):
            print("文件不存在！")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        results = predict_messages(lines)
    else:
        print("无效选项！")
        sys.exit(1)

    # 输出结果
    print("\n=== 预测结果 ===")
    for msg, label, prob in results:
        print(f"短信: {msg}")
        print(f"预测: {label} (Spam 概率={prob:.4f})")
        print("-" * 50)