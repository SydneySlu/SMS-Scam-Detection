# SMS Spam Detection

一个完整的短信垃圾分类系统，支持两种模型：
- **Logistic Regression (Scratch)**: 从零实现的逻辑回归
- **Logistic Regression (Sklearn)**: 使用 sklearn 的逻辑回归

## 项目结构

```text
├── src/
│   ├── data/                 # 数据集 (spam.csv)
│   ├── models/               # 保存的模型
│   ├── results/              # 可视化输出 (混淆矩阵/ROC 曲线)
│   ├── data_preprocessing.py # 数据预处理
│   ├── model_from_scratch.py # 手写逻辑回归
│   ├── model_sklearn.py      # sklearn 逻辑回归
│   ├── evaluation.py         # 模型评估与可视化
│   ├── predict_friendly.py   # sklearn 模型预测
│   ├── predict_with_scratch.py # scratch 模型预测
│   ├── test_all.py           # 一键运行并对比两模型
│   ├── demo.ipynb            # 交互式 Notebook
│   └── main.py               # 主入口
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
````

## 快速开始

### 1\. 环境准备

```bash
pip install -r requirements.txt
```

### 2\. 训练模型

```bash
python src/main.py
```

### 3\. 模型对比与测试

```bash
python src/test_all.py
```

## 模型预测

**sklearn 模型：**

```bash
python src/predict_friendly.py
```

**scratch 模型：**

```bash
python src/predict_with_scratch.py
```

## 交互式演示

```bash
jupyter notebook src/demo.ipynb
```

## License

本项目遵循 MIT License。
