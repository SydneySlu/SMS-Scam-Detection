"""
main.py
功能：程序入口，整合数据处理、模型训练、评估
"""

from src.data_preprocessing import load_and_preprocess_data
from src.model_from_scratch import LogisticRegressionScratch
from src.model_sklearn import train_sklearn_model, predict_sklearn_model
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
import joblib
import os

def main():
    # 1. 加载并预处理数据
    filepath = "data/spam.csv"
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data(filepath)

    # 2. 手写逻辑回归
    scratch_model = LogisticRegressionScratch(lr=0.1, n_iter=300,class_weight="balanced", verbose=True)
    scratch_model.fit(X_train, y_train)
    y_pred_scratch = scratch_model.predict(X_test)
    evaluate_model(y_test, y_pred_scratch, "Logistic Regression (Scratch)")

    # 3. sklearn 逻辑回归
    sklearn_model = train_sklearn_model(X_train, y_train)
    y_pred_sklearn = predict_sklearn_model(sklearn_model, X_test)
    evaluate_model(y_test, y_pred_sklearn, "Logistic Regression (Sklearn)")

    # 4. 可视化
    plot_confusion_matrix(y_test, y_pred_sklearn, save_path="results/confusion_matrix.png")
    plot_roc_curve(sklearn_model, X_test, y_test, save_path="results/roc_curve.png")

    # 保存向量器和 sklearn 模型
    os.makedirs("models", exist_ok=True)

    # 保存向量器
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # 保存 sklearn 模型
    joblib.dump(sklearn_model, "models/sklearn_model.pkl")

    # 保存手写模型
    joblib.dump(scratch_model, "models/scratch_model.pkl")

    joblib.dump({"weights": scratch_model.weights, "bias": scratch_model.bias}, "models/scratch_model.pkl")

    print("[INFO] 已保存 vectorizer 和模型到 models/ 文件夹")

if __name__ == "__main__":
    main()
