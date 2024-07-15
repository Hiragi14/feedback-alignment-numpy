import numpy as np
# np.random.seed(34)
np.random.seed(42)
import numpy as np

# def generate_independent_matrix(rows, cols):
#     # ランダムな正則行列 P を生成する
#     P = np.random.uniform(low=-0.08, high=0.08, size=(rows, rows))
#     while np.linalg.matrix_rank(P) < rows:
#         P = np.random.uniform(low=-0.08, high=0.08, size=(rows, rows))
    
#     # ランダムな対角行列 D を生成する
#     D = np.diag(np.random.rand(rows))
    
#     # 行列 A を生成する
#     A = np.dot(P, np.dot(D, np.linalg.inv(P)))
    
#     # 必要に応じて行列 A を cols 列までトリミングする
#     if A.shape[1] > cols:
#         A = A[:, :cols]
    
#     return A

def generate_independent_matrix(rows, cols):
    # ランダムな正則行列 P を生成する
    P = np.random.uniform(low=-1, high=1, size=(rows, rows))
    while np.linalg.matrix_rank(P) < rows:
        P = np.random.uniform(low=-1, high=1, size=(rows, rows))
    
    # ランダムな対角行列 D を生成する
    D = np.diag(np.random.rand(rows))
    
    # 行列 A を生成する
    A = np.dot(P, np.dot(D, np.linalg.inv(P)))
    
    # 必要に応じて行列 A を cols 列までトリミングする
    if A.shape[1] > cols:
        A = A[:, :cols]
    
    # 行列 A の要素を [-0.08, 0.08] の範囲にスケーリングする
    max_abs_value = np.max(np.abs(A))
    A = A * (0.08 / max_abs_value)
    
    return A

# FAの全結合層
class Dense_FA:
    def __init__(self, in_dim, out_dim, function, deriv_function) -> None:
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float64')
        # self.W = np.zeros((in_dim, out_dim)).astype('float64')
        self.B = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float64')
        # self.B = generate_independent_matrix(in_dim, out_dim).astype('float64')
        self.b = np.zeros(out_dim).astype('float64')
        self.function = function
        self.deriv_function = deriv_function
        self.x = None
        self.u = None
        self.dW = None
        self.db = None
        self.delta_backprop = None
    def __call__(self, x):
        self.x = x
        self.u = self.x @ self.W + self.b
        h = self.function(self.u)
        return h
    def feedback(self, delta, B):
        self.delta = self.deriv_function(self.u) * (delta @ B.T)
        return self.delta
    def compute_grad(self) -> None:
        batch_size = self.delta.shape[0]
        self.dW = np.matmul(self.x.T, self.delta) / batch_size
        self.db = np.matmul(np.ones(batch_size), self.delta) / batch_size


# 全結合層
class Dense:
    def __init__(self, in_dim, out_dim, function, deriv_function) -> None:
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype("float64")
        # self.W = np.zeros((in_dim, out_dim)).astype('float64')
        self.b = np.zeros(out_dim).astype("float64")

        self.function = function
        self.deriv_function = deriv_function

        self.x = None
        self.u = None
        self.dW = None
        self.db = None

        self.params_idxs = np.cumsum([self.W.size, self.b.size])
    def __call__(self, x):
        self.x = x
        self.u = np.matmul(self.x, self.W) + self.b
        h = self.function(self.u)
        return h
    def b_prop(self, delta, W):
        self.delta = self.deriv_function(self.u) * np.matmul(delta, W.T)
        return self.delta
    def compute_grad(self) -> None:
        batch_size = self.delta.shape[0]
        self.dW = np.matmul(self.x.T, self.delta) / batch_size
        self.db = np.matmul(np.ones(batch_size), self.delta) / batch_size