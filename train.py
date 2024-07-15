from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

import numpy as np
import pandas as pd
from lib.functions import np_log, relu, deriv_relu, deriv_softmax, softmax
from lib.model import Model, Model_FA
from lib.datahandler import create_batch

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

# 現在の日時を取得してフォルダ名を生成
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/mlp_fa_{current_time}'

# TensorBoard writerの初期化
writer = SummaryWriter(log_dir)

# np.random.seed(34)
np.random.seed(42)

# MNISTデータの取得
digits = fetch_openml(name='mnist_784', version=1)
x_mnist = np.array(digits.data)
t_mnist = np.array(digits.target)

x_mnist = x_mnist.astype("float64") / 255.  # 値を[0, 1]に正規化する
t_mnist = np.eye(N=10)[t_mnist.astype("int32").flatten()]  # one-hotベクトルにする

x_mnist = x_mnist.reshape(x_mnist.shape[0], -1)  # 1次元に変換

# train data: 5000, valid data: 10000 test data: 10000にする
x_train_mnist, x_test_mnist, t_train_mnist, t_test_mnist =\
    train_test_split(x_mnist, t_mnist, test_size=10000)
x_train_mnist, x_valid_mnist, t_train_mnist, t_valid_mnist =\
    train_test_split(x_train_mnist, t_train_mnist, test_size=10000)

def train(model, x, t, eps=0.01):
    y = model(x)
    delta = y - t
    model.backward(delta)
    model.update(eps)
    cost = (-t * np_log(y)).sum(axis=1).mean()
    return cost

def valid(model, x, t):
    y = model(x)
    cost = (-t * np_log(y)).sum(axis=1).mean()
    return cost, y

model_FA = Model_FA(
    hidden_dims=[784, 100, 100, 10], 
    # hidden_dims=[784, 100, 10], 
    activate_function=[relu, relu, softmax], 
    deriv_activate_function=[deriv_relu, deriv_relu, deriv_softmax])

model = Model(
    hidden_dims=[784, 100, 100, 10], 
    # hidden_dims=[784, 100, 10], 
    activation_functions=[relu, relu, softmax], 
    deriv_functions=[deriv_relu, deriv_relu, deriv_softmax])


batch_size = 128
epoch = 50
lr =0.1

Accuracy_array = np.array([])
Accuracy_array_fa = np.array([])
Cost_array = np.array([])
Cost_array_fa = np.array([])

for i in range(epoch):
    x_train_mnist, t_train_mnist = shuffle(x_train_mnist, t_train_mnist)
    x_train_batch, t_train_batch = \
        create_batch(x_train_mnist, batch_size), create_batch(t_train_mnist, batch_size)
    
    for x, t in zip(x_train_batch, t_train_batch):
        cost_FA = train(model_FA, x, t, eps=lr)
        cost = train(model, x, t, eps=lr)
    cost_FA, y_pred_FA = valid(model_FA, x_valid_mnist, t_valid_mnist)
    cost, y_pred = valid(model, x_valid_mnist, t_valid_mnist)
    accuracy_FA = accuracy_score(t_valid_mnist.argmax(axis=1), y_pred_FA.argmax(axis=1))
    accuracy = accuracy_score(t_valid_mnist.argmax(axis=1), y_pred.argmax(axis=1))

    for j, layer in enumerate(model.layers):
        writer.add_histogram(f"layer_{j+1}_weights/BP", layer.W.flatten(), i, bins='auto')
    for j, layer in enumerate(model_FA.layers):
        writer.add_histogram(f"layer_{j+1}_weights/FA", layer.W.flatten(), i, bins='auto')
        writer.add_histogram(f"layer_{j+1}_fixed_matrix/FA", layer.B.flatten(), i, bins='auto')

    writer.add_scalar("Loss/BP", cost, epoch * batch_size + i)
    writer.add_scalar("Loss/FA", cost_FA, epoch * batch_size + i)
    writer.add_scalar("Accuracy/BP", accuracy, epoch * batch_size + i)
    writer.add_scalar("Accuracy/FA", accuracy_FA, epoch * batch_size + i)

    print(f"#####EPOCH: {i+1}")
    print(f"BP Valid[Cost: {cost:.3f}, Accuracy: {accuracy:.3f}]")
    print(f"FA Valid[Cost: {cost_FA:.3f}, Accuracy: {accuracy_FA:.3f}]")
    Accuracy_array = np.append(Accuracy_array, accuracy)
    Accuracy_array_fa = np.append(Accuracy_array_fa, accuracy_FA)
    Cost_array = np.append(Cost_array, cost)
    Cost_array_fa = np.append(Cost_array_fa, cost_FA)

#data = pd.Series(data=Accuracy_array, index=x, name='Accuracy_BP')
x = range(epoch)
Accuracy_array = pd.Series(data=Accuracy_array, index=x, name='Accuracy_BP')
Accuracy_array_fa = pd.Series(data=Accuracy_array_fa, index=x, name='Accuracy_FA')
Cost_array = pd.Series(data=Cost_array, index=x, name='Cost_BP')
Cost_array_fa = pd.Series(data=Cost_array_fa, index=x, name='Cost_FA')

data = pd.concat([Accuracy_array ,Accuracy_array_fa, Cost_array, Cost_array_fa], axis=1)

# data.to_csv('./data/data_fa.csv')