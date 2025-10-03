"""
model_from_scratch.py
功能：更健壮的手写 Logistic Regression
支持：稀疏矩阵、样本权重 / class_weight='balanced'、mini-batch、L2 正则、loss 记录
"""

import numpy as np
from scipy import sparse

class LogisticRegressionScratch:
    # def __init__(
    #     self,
    #     lr=0.01,
    #     n_iter=1000,
    #     reg_lambda=0.0,
    #     batch_size=None,
    #     class_weight=None,
    #     tol=None,
    #     verbose=False,
    #     random_state=None,
    #     lr_decay=0.0,
    # ):
    def __init__(
            self,
            lr=0.1,  # 提高学习率
            n_iter=300,  # 增加迭代次数
            reg_lambda=0.0,  # 先关掉正则，避免梯度冲掉
            batch_size=None,
            class_weight="balanced",  # 默认加平衡
            tol=1e-4,  # 增加收敛判定
            verbose=True,  # 打印 loss
            random_state=None,
            lr_decay=0.0,
    ):
        """
        参数说明:
          lr: 学习率
          n_iter: 迭代次数（epoch）
          reg_lambda: L2 正则强度
          batch_size: None 表示全量 batch；>0 表示 mini-batch 大小
          class_weight: None 或 'balanced' 或 dict {0: w0, 1: w1}
          tol: 若提供，loss 收敛到 tol 则提前停止
          lr_decay: 每个 epoch 学习率衰减项（lr = lr / (1 + lr_decay * epoch)）
        """
        self.lr = lr
        self.n_iter = int(n_iter)
        self.reg_lambda = float(reg_lambda)
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.lr_decay = lr_decay

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _is_sparse(self, X):
        return sparse.issparse(X)

    def sigmoid(self, z):
        z = np.asarray(z)
        # 数值稳定实现
        pos_mask = z >= 0
        out = np.empty_like(z, dtype=float)
        out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        neg_mask = ~pos_mask
        if neg_mask.any():
            exp_z = np.exp(z[neg_mask])
            out[neg_mask] = exp_z / (1.0 + exp_z)
        return out

    def _compute_loss(self, y, y_pred, sample_weight=None):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # element-wise loss
        elem = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        if sample_weight is None:
            loss = elem.mean()
        else:
            loss = np.average(elem, weights=sample_weight)
        if self.reg_lambda > 0 and (self.weights is not None):
            loss += (self.reg_lambda / (2.0 * len(y))) * np.sum(self.weights ** 2)
        return float(loss)

    def _init_params(self, n_features):
        if self.weights is None:
            self.weights = np.zeros(n_features, dtype=float)
        if self.bias is None:
            self.bias = 0.0

    def _make_sample_weight(self, y):
        # returns array of shape (n_samples,)
        if self.class_weight is None:
            return None
        y_int = np.asarray(y).astype(int)
        n_samples = len(y_int)
        classes, counts = np.unique(y_int, return_counts=True)
        class_count_map = dict(zip(classes.tolist(), counts.tolist()))
        if self.class_weight == "balanced":
            n_classes = len(classes)
            weights = {c: n_samples / (n_classes * class_count_map[c]) for c in classes}
        elif isinstance(self.class_weight, dict):
            weights = self.class_weight
        else:
            return None
        # map to sample weights
        sw = np.array([weights[int(yi)] for yi in y_int], dtype=float)
        return sw

    def fit(self, X, y):
        """
        X: csr_matrix or ndarray, shape (n_samples, n_features)
        y: array-like {0,1}
        """
        # prepare
        rng = np.random.RandomState(self.random_state)
        y = np.asarray(y).reshape(-1).astype(float)
        n_samples, n_features = X.shape
        self._init_params(n_features)
        sample_weight_full = self._make_sample_weight(y)  # may be None

        # mini-batch setup
        if self.batch_size is None:
            batch_size = n_samples
        else:
            batch_size = int(self.batch_size)

        self.loss_history = []
        lr = float(self.lr)

        for epoch in range(self.n_iter):
            # optionally decay lr
            if self.lr_decay:
                lr = self.lr / (1.0 + self.lr_decay * epoch)

            # shuffle indices for minibatch SGD
            indices = np.arange(n_samples)
            if batch_size < n_samples:
                rng.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                # slice X,y
                if self._is_sparse(X):
                    Xb = X[batch_idx]
                else:
                    Xb = X[batch_idx, :]

                yb = y[batch_idx]
                swb = None if sample_weight_full is None else sample_weight_full[batch_idx]

                # forward: linear and sigmoid
                linear = Xb.dot(self.weights) + self.bias
                # ensure 1d numpy
                if sparse.issparse(linear):
                    linear = linear.toarray().ravel()
                else:
                    linear = np.asarray(linear).ravel()
                y_pred = self.sigmoid(linear)

                # error and weighted gradient
                err = y_pred - yb  # shape (batch_size,)
                if swb is not None:
                    weighted_err = err * swb
                    denom = swb.sum()
                    if denom == 0:
                        denom = len(swb)
                else:
                    weighted_err = err
                    denom = len(err)

                # grad_w = Xb.T.dot(weighted_err) / denom
                grad_w = Xb.T.dot(weighted_err)
                # ensure grad_w is 1d numpy
                if sparse.issparse(grad_w):
                    grad_w = grad_w.toarray().ravel()
                else:
                    grad_w = np.asarray(grad_w).ravel()
                grad_w = grad_w / denom

                grad_b = weighted_err.sum() / denom

                # # L2
                # if self.reg_lambda > 0:
                #     grad_w = grad_w + (self.reg_lambda / batch_size) * self.weights
                if self.reg_lambda > 0:
                    grad_w = grad_w + (self.reg_lambda / denom) * self.weights

                # update
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b

            # epoch end: compute full-loss for monitoring (using possibly full-sample weights)
            linear_full = X.dot(self.weights) + self.bias
            linear_full = np.asarray(linear_full).ravel()
            y_pred_full = self.sigmoid(linear_full)
            loss = self._compute_loss(y, y_pred_full, sample_weight=sample_weight_full)
            self.loss_history.append(loss)

            if self.verbose and (epoch % max(1, self.n_iter // 10) == 0):
                print(f"[epoch {epoch}] loss={loss:.6f}")

            if (self.tol is not None) and epoch > 0:
                if abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                    if self.verbose:
                        print(f"Converged at epoch {epoch}, loss delta < tol")
                    return self

        return self

    def predict_proba(self, X):
        linear = X.dot(self.weights) + self.bias
        if sparse.issparse(linear):
            linear = linear.toarray().ravel()
        else:
            linear = np.asarray(linear).ravel()
        return self.sigmoid(linear)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
