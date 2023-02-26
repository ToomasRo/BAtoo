import random
import numpy as np
import tensorflow as tf

from scripts import Slopes
import matplotlib.pyplot as plt


def reset_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def joonista(m, X, y, bpoint_fn=Slopes.breakpoint_finder, max_delta=0.0001):
    X_pred = np.linspace(min(X)-0.2, max(X)+0.2, 1000)
    X_vahemik = X

    bpoints = bpoint_fn(m, X_pred, max_delta)

    if len(bpoints) != 0 and isinstance(bpoints[0][1], np.ndarray):
        bpoints = [bp[0] for bp in bpoints]
    print(f"model contains {len(bpoints)} breaks")

    _fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
    ax1.scatter(X, y, marker="+", color='red', label="target")
    ax1.plot(X_pred, m.predict(X_pred), label="prediction",  linewidth=0.5)
    if len(bpoints) != 0:
        ax1.scatter(*zip(*bpoints), marker="x",
                    color='magenta', label="breaks")
        ax1.legend()
    ax2.scatter(X, y, marker="+", color='red', label="target")
    plt.show()
    return _fig, (ax1, ax2)


def joonista_temp(m, X, y, bpoint_fn=Slopes.breakpoint_finder, max_delta=0.0001):
    X_pred = np.linspace(-1.2, 1.2, 1000)
    X_vahemik = X

    bpoints = bpoint_fn(m, X_vahemik, max_delta)

    if len(bpoints) != 0 and isinstance(bpoints[0][1], np.ndarray):
        bpoints = [(bp[0][0], bp[0][1]/1000) for bp in bpoints]
    print(f"model contains {len(bpoints)} breaks")

    _fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
    ax1.scatter(X, y, marker="+", color='red', label="target")
    ax1.plot(X_pred, m.predict(X_pred)/1000,
             label="prediction",  linewidth=0.5)
    if len(bpoints) != 0:
        ax1.scatter(*zip(*bpoints), marker="x",
                    color='magenta', label="breaks")
        ax1.legend()
    ax2.scatter(X, y, marker="+", color='red', label="target")
    plt.show()
    return _fig, (ax1, ax2)


def joonista_variance(m: tf.keras.Sequential, X_test, X_train, y_train=None, bpoint_fn=Slopes.breakpoint_finder, *, ground_truth=False, xlim=None, ylim=None, returnplt=False):

    # for each prediction, plot the mean value and the variance of the prediction
    if y_train is None:
        y_train = X_train**2

    bpoints = bpoint_fn(m, X_test)
    #print(bpoints)

    # if len(bpoints)!=0 and isinstance(bpoints[0][1], np.ndarray):
    #     print("Muudame bpointe")
    #     bpoints = [(bp[0][0], bp[0][1]/1000) for bp in bpoints]
    # print(bpoints)
    _patterns = [bp[1] for bp in bpoints]
    bpoints = [bp[0] for bp in bpoints]
    # print(bpoints)
    # print(patterns)

    print(f"model contains {len(bpoints)} breaks")

    y_pred = m.predict(X_test)
    # print(y_pred)
    y_pred_mean, y_pred_logvar = y_pred[:, 0], y_pred[:, 1]
    y_pred_var = np.exp(y_pred_logvar)


    fig, ax = plt.subplots(figsize=(6, 6))
    plt.xlim(xlim)
    plt.ylim(ylim)

    # absoluutselt õige
    if ground_truth:
        plt.plot(np.linspace(-1,1,1000), ground_truth(np.linspace(-1,1,1000)), label="õige", alpha=0.5, color="yellow")
    
    
    plt.plot(X_test, y_pred_mean, label="mean")
    plt.fill_between(X_test, y_pred_mean - y_pred_var,
                     y_pred_mean + y_pred_var, alpha=0.2, label="variance")
    plt.scatter(X_train, y_train, marker='+', color="red", label="train")

    temp_bpoints = list(zip(*bpoints))
    bx, by = temp_bpoints[0], temp_bpoints[1]

    plt.scatter(bx, by, marker="o", color="green", label="breaks")

    plt.legend()

    if returnplt:
        return plt
    
    plt.show()


def neg_log_likelihood(y_true, y_predicted):

    y_pred_mean, y_pred_var = y_predicted[:, 0:1], y_predicted[:, 1:]

    # y_pred_var ongi log variance
    return tf.reduce_mean(y_pred_var + tf.math.square((y_pred_mean - y_true) / tf.math.exp(y_pred_var)))

def neg_log_likelihood2(ytrue ,ypreds):
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*tf.reduce_sum(tf.math.square((ytrue-mu)/tf.math.exp(logsigma)),axis=1)
    sigma_trace = -tf.reduce_sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return tf.reduce_mean(-log_likelihood)


def train_data_maker(elements:list[tuple[float, float, float, float]], fn=lambda x: x**2, seed=0) -> tuple[np.array, np.array]:
    """Loob treeningandmed, võtab sisse listi, kus iga element on tuple:
        - mean
        - std
        - n_samples
        - noise_std

    :param elements: list of tuples, where each tuple is (mean, std, n_samples, noise_std)
    :param fn: f(x), funktsioon, defaults to lambdax:x**2
    :return: treeningandmed
    """    
    reset_seeds(seed)


    X, y = [], []
    for el in elements:
        xi = np.random.normal(el[0], el[1], el[2])
        yi = fn(xi) + np.random.normal(0, el[3], size=el[2])
        X.extend(xi)
        y.extend(yi)
    X, y = np.array(X), np.array(y)
    return X, y
