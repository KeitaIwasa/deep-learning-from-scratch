# coding: utf-8
import os
import sys
import pickle

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam

print("Loading MNIST data...")
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
t_train = t_train[:300]

print("Creating network...")
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=0)
optimizer = Adam(lr=0.01)

print("Training model...")
max_epochs = 50  # 少し短めに
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(f"epoch: {epoch_cnt}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

print("Saving model...")
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(network, f)
print("Model saved as 'trained_model.pkl'")