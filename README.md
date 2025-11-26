# SMS Spam Detection

一个完整的 **短信垃圾分类系统**，支持两种模型：
- **Logistic Regression (Scratch)**: 从零实现的逻辑回归
- **Logistic Regression (Sklearn)**: 使用 sklearn 的逻辑回归

## 📂 项目结构
├── src/
│ ├── data/ # 数据集 (spam.csv)
│ ├── models/ # 保存的模型
│ ├── results/ # 可视化输出 (混淆矩阵/ROC 曲线)
│ ├── data_preprocessing.py # 数据预处理
│ ├── model_from_scratch.py # 手写逻辑回归
│ ├── model_sklearn.py # sklearn 逻辑回归
│ ├── evaluation.py # 模型评估与可视化
│ ├── predict_friendly.py # sklearn 模型预测
│ ├── predict_with_scratch.py # scratch 模型预测
│ ├── test_all.py # 一键运行并对比两模型
│ ├── demo.ipynb # 交互式 Notebook
│ └── main.py # 主入口
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt

python src/main.py

python test_all.py

sklearn 模型：
python predict_friendly.py

scratch 模型：
python predict_with_scratch.py

jupyter notebook demo.ipynb
