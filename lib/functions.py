import numpy as np

def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))

# 活性化関数とその微分の定義
def sigmoid(x):
    # x >= 0: 1 / (1 + exp(-x))
    # x <  0: exp(x) / (1 + exp(x))
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return 1 - tanh(x) ** 2

def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def deriv_softmax(x):
    return softmax(x) * (1 - softmax(x))