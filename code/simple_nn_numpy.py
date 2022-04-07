# To create plots used these versions and Python 3.10.3:
# matplotlib==3.5.1
# numpy==1.22.3

from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt

def linear(inp, w, b):
    return inp @ w + b

def linear_backward(inp, w, b, dout):
    db = dout.mean(axis=0)
    dw = inp.T @ dout
    dinp = dout @ w.T
    return dinp, dw, db

def relu(inp):
    return np.maximum(0, inp)

def relu_backward(inp, dout):
    return (inp > 0) * dout

def mse(inp, true):
    return np.square(inp - true).mean()

def mse_backward(inp, true):
    return (inp - true) * (2 / np.prod(inp.shape))

class WeightInit(IntEnum):
    simple=0
    kaiming=1

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, out_dim, weight_init=WeightInit.simple):
        self.w1, self.b1 = self._get_weights(input_dim, hidden_dim, weight_init)
        self.w2, self.b2 = self._get_weights(hidden_dim, out_dim, weight_init)

    def _get_weights(self, input_dim, output_dim, weight_init):
        scale = 1.0
        if weight_init == WeightInit.kaiming:
            scale = np.sqrt(2 / input_dim)
        w = np.random.normal(size=(input_dim, output_dim), scale=scale)
        b = np.zeros(output_dim)
        return w, b
    
    def forward(self, inp):
        self.inp = inp
        self.linear1 = linear(self.inp, self.w1, self.b1)
        self.relu1 = relu(self.linear1)
        self.linear2 = linear(self.relu1, self.w2, self.b2)
        return self.linear2

    def backward(self, dlinear2):
        drelu1, self.dw2, self.db2 = linear_backward(self.relu1, self.w2, self.b2, dlinear2)
        dlinear1 = relu_backward(self.linear1, drelu1)
        dinp, self.dw1, self.db1 = linear_backward(self.inp, self.w1, self.b1, dlinear1)

    def sgd_update(self, lr):
        self.w1 -= self.dw1 * lr
        self.b1 -= self.db1 * lr
        self.w2 -= self.dw2 * lr
        self.b2 -= self.db2 * lr

    def _print_mean_and_var(self, X):
        y = model.forward(X)
        print_stats("input", X)
        print_stats("layer1", model.relu1)
        print_stats("layer2", model.linear2)


def noop(x):
    return x

def square(X, w, b):
    return linear(np.square(X), w, b)

def wave(X, w, b):
    return linear(np.sin(X), w, b)

def get_data(n, input_features, output_features, transform, x_spread=3.0, noise_spread=4.0):
    w = np.random.normal(loc=0.0, scale=3.0, size=(input_features, output_features))
    b = np.random.normal(loc=0.0, scale=5.0)
    noise = np.random.normal(size=(n, output_features), scale=noise_spread)
    X = np.random.normal(scale=x_spread, size=(n, input_features))
    y = transform(X, w, b) + noise
    return X, y, w, b

def get_linear_data(n, input_features, output_features):
    return get_data(n, input_features, output_features, linear)

def get_square_data(n, input_features, output_features):
    return get_data(n, input_features, output_features, square)

def get_wave_data(n, input_features, output_features):
    return get_data(n, input_features, output_features, wave, 5.0, 0.3)


def print_stats(name, X):
    print(f"After {name}: mean={X.mean():.2f}, variance={X.var():.3f}")

def print_final_loss(y_train, y_val, train_loss, val_loss):
        avg_train = y_train.mean()
        predict_avg_loss_train = mse(avg_train, y_train)
        predict_avg_loss_val = mse(avg_train, y_val)
        print(f"Train loss={train_loss:.4f}, val. loss={val_loss:.4f}")
        print(f"Using avg. response: train loss={predict_avg_loss_train:.4f}, val. loss={predict_avg_loss_val:.4f}")

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(16, 9))
    plt.plot(range(len(train_losses)), train_losses, label="Train loss")
    plt.plot(range(len(val_losses)), val_losses, label="Val. loss")
    plt.yscale("log")
    plt.legend()
    if save_plots: plt.savefig(f"loss_for_{name}")
    if show_plots: plt.show()

def plot_data(X, X_val, y_val, X_train, y_train, true_data_func, model):
        X = np.sort(X, axis=0)
        plt.figure(figsize=(16, 9))
        plt.scatter(X_val, y_val, color="blue", label="Val. data")
        plt.scatter(X_train, y_train, color="black", label="Train data")
        plt.plot(X, true_data_func(X), color="black", label="True function")
        plt.scatter(X_val, model.forward(X_val), color="orange", label="Predicted for val. data")
        plt.plot(X, model.forward(X), color="orange", label="Model function")
        plt.legend()
        if save_plots: plt.savefig(f"data_for_{name}")
        if show_plots: plt.show()


data_types = [("linear", get_linear_data, linear), ("square", get_square_data, square), ("wave", get_wave_data, wave)]
np.random.seed(0)

# Config
show_plots = False
save_plots = True
print_mean_and_var = False
normalize_input = False # TODO: breaks plotting the true function
weight_init = WeightInit.simple
n = 200
inp_dim = 1
out_dim = 1
train_split = 0.8
hidden_dim = 20
lr = 0.5
updates = 10000

train_size = int(n * train_split)
val_size = n - train_size
print(f"Train size = {train_size}, validation size = {val_size}")

for (name, get_data_func, data_func) in data_types:
    print(f"\n_______________\nFitting {name} data\n_______________")
    X, y, true_w, true_b = get_data_func(n, inp_dim, out_dim)
    true_data_func = lambda X: data_func(X, true_w, true_b)
    if normalize_input: X = (X - X.mean()) / X.std()
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    model = SimpleNN(inp_dim, hidden_dim, out_dim, weight_init)
    if print_mean_and_var: model._print_mean_and_var(X_train)
        
    train_losses, val_losses = [], []
    for i in range(updates):
        train_predicted = model.forward(X_train)
        train_loss = mse(train_predicted, y_train)
        
        model.backward(train_loss)
        model.sgd_update(lr * (1 - i / updates))

        val_predicted = model.forward(X_val)
        val_loss = mse(val_predicted, y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if i % 200 == 0:
            print(f"Update {i}/{updates}, train loss={train_loss:.4f}, val. loss={val_loss:.4f}", end="        \r")

    print()
    print_final_loss(y_train, y_val, train_loss, val_loss)
    plot_losses(train_losses, val_losses)

    if inp_dim == 1 and out_dim == 1:
        plot_data(X, X_val, y_val, X_train, y_train, true_data_func, model)
