# test_all.py
import os
import joblib
from src.data_preprocessing import load_and_preprocess_data
from src.model_from_scratch import LogisticRegressionScratch
from src.model_sklearn import train_sklearn_model, predict_sklearn_model
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

def run_demo():
    os.makedirs("results", exist_ok=True)

    # 1. 加载数据
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data("data/spam.csv")

    # 2. Scratch 模型
    scratch_model = LogisticRegressionScratch(lr=0.1, n_iter=300, class_weight="balanced", verbose=False)
    scratch_model.fit(X_train, y_train)
    y_pred_scratch = scratch_model.predict(X_test)
    metrics_scratch = evaluate_model(y_test, y_pred_scratch, "Scratch LR")
    plot_confusion_matrix(y_test, y_pred_scratch, save_path="results/confusion_scratch.png")
    plot_roc_curve(scratch_model, X_test, y_test, save_path="results/roc_scratch.png")

    # 3. Sklearn 模型
    sklearn_model = train_sklearn_model(X_train, y_train)
    y_pred_sklearn = predict_sklearn_model(sklearn_model, X_test)
    metrics_sklearn = evaluate_model(y_test, y_pred_sklearn, "Sklearn LR")
    plot_confusion_matrix(y_test, y_pred_sklearn, save_path="results/confusion_sklearn.png")
    plot_roc_curve(sklearn_model, X_test, y_test, save_path="results/roc_sklearn.png")

    # 4. 保存结果
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(scratch_model, "models/scratch_model.pkl")
    joblib.dump(sklearn_model, "models/sklearn_model.pkl")

    print("\n=== 对比结果 ===")
    print("Scratch LR:", metrics_scratch)
    print("Sklearn LR:", metrics_sklearn)
    print("\n可视化图表保存在 results/ 文件夹")

if __name__ == "__main__":
    run_demo()