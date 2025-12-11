# SMS Spam Detection

这是一个完整的**短信垃圾分类系统 (SMS Spam Detection System)**。该项目旨在对比和演示机器学习模型在垃圾短信识别任务中的应用，支持从零实现与使用成熟框架两种方式。

## 主要功能

* **双模型支持**：
    * **Logistic Regression (Scratch)**: 不依赖机器学习库，完全从零手写的逻辑回归算法，用于理解底层原理。
    * **Logistic Regression (Sklearn)**: 使用 `scikit-learn` 库实现的工业级逻辑回归。
* **模型评估**: 自动生成混淆矩阵 (Confusion Matrix) 和 ROC 曲线，对比模型性能。
* **交互式演示**: 提供 Jupyter Notebook 用于交互式数据分析和测试。
* **一键测试**: 支持一键运行并对比两个模型的表现。

---

## 项目结构

```text
├── src/
│   ├── data/                 # 数据集存储 (包含 spam.csv)
│   ├── models/               # 训练好的模型保存目录
│   ├── results/              # 可视化输出 (图表/评估结果)
│   ├── data_preprocessing.py # 数据清洗与预处理模块
│   ├── model_from_scratch.py # 手写逻辑回归模型实现
│   ├── model_sklearn.py      # Sklearn 逻辑回归模型实现
│   ├── evaluation.py         # 评估与绘图工具
│   ├── predict_friendly.py   # 使用 Sklearn 模型进行预测
│   ├── predict_with_scratch.py # 使用 Scratch 模型进行预测
│   ├── test_all.py           # 一键测试脚本
│   ├── demo.ipynb            # 交互式演示 Notebook
│   └── main.py               # 项目主入口 (训练与保存)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

## 快速开始
1. 环境准备

```bash
pip install -r requirements.txt

2. 训练模型

```bash
python src/main.py

3. 模型对比与测试

```bash
python src/test_all.py

## 模型测试

```bash
python src/predict_friendly.py

```bash
python src/predict_with_scratch.py

## 交互式演示

```bash
jupyter notebook src/demo.ipynb

##License
本项目遵循 MIT License。


