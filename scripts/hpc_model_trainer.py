import sys
import os

print(os.getcwd())
sys.path.append(os.getcwd())

import numpy as np
import random
import tensorflow as tf

from tensorflow import keras
from keras import layers

from matplotlib import pyplot as plt
import seaborn as sns

from scripts.CustomCallbacks import CustomLogger
import scripts.utils as utils
import scripts.Slopes as Slopes

from tqdm import tqdm
import json
utils.reset_seeds(0)


def train_model(
    trainX: np.ndarray,
    nn_size: tuple[int, int] = (20, 20),
    optimizer: keras.optimizers = keras.optimizers.Adam(
        learning_rate=0.003, amsgrad=True, epsilon=1e-3),
    epochs: int = 1000,
    batch_size: int = None,
    fn=lambda x: 0*x,
    noise_fn=lambda x: 0.3 *
        np.random.randn(len(x)) + 0.3 * x * np.random.randn(len(x)),
    reverse_fn: bool = False,
    reverse_noise: bool = False,
    seed: int = None
) -> tuple[keras.Model, keras.callbacks.History, tuple[np.ndarray, np.ndarray]]:
    """Mudeli treenimine, võib kasutada erinevaid funktsioone ja müra, tagastab mudeli, ajaloo ja treeningandmed

    :param trainX: treeningX andmed
    :param nn_size: võrgu kihtide suurused, hetkel ainult 2 kihti lubatud, defaults to (20, 20)
    :param optimizer: optimiseerija, hetkel valitud hea väiksematele andmehulkadele, defaults to keras.optimizers.Adam( learning_rate=0.003, amsgrad=True, epsilon=1e-3)
    :param epochs: epochide kogus, defaults to 1000
    :param batch_size: batch size, kasutab valemit min(16, kahe aste mis > len(X)/5), defaults to 16
    :param fn: funktsioon mida õppida, defaults to lambdax:0*x
    :param noise_fn: mürafn, defaults to lambdax:0.3*np.random.randn(len(x))+0.3*x*np.random.randn(len(x))
    :param reverse_fn: kas pöörata fn tagurpidi, defaults to False
    :param reverse_noise: kas pöörata müra tagurpidi, defaults to False
    :param seed: seed, defaults to 42
    :return: treenitud mudel, ajalugu, treeningandmed
    """

    utils.reset_seeds(seed)

    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(units=nn_size[0], activation='relu', name="layer_1"),
        layers.Dense(units=nn_size[1], activation='relu', name="layer_2"),
        layers.Dense(units=2, activation='linear', name="layer_3")
    ])

    X = trainX

    # kõik võimalikud kombod reverse_fn ja reverse_noise. Default on tavalist pidi
    if reverse_fn and reverse_noise:
        y = fn(X[::-1]) + noise_fn(X[::-1])
    elif reverse_fn:
        y = fn(X[::-1]) + noise_fn(X)
    elif reverse_noise:
        y = fn(X) + noise_fn(X[::-1])
    else:
        y = fn(X) + noise_fn(X)

    X_train, y_train = X, y

    model.compile(
        optimizer=optimizer,
        loss=utils.neg_log_likelihood,
    )

    if batch_size is None:
        batch_size = min(16, max(2, 2**int(np.log2(len(trainX)/5+0.001)+1)))

    # TODO: validation_batch_size ja validation_freq saab kasutada et vaadata kuidas viga treeningu jooksul muutub
    h = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                  callbacks=[CustomLogger(100)], shuffle=True, )
    return model, h, (X_train, y_train)


def fn(X):
    return 0*X


def noise_fn(X): 
    return 0.3 * np.random.randn(len(X)) + 0.3 * X * np.random.randn(len(X))



def save_everything(directory, train_size, seed, y_pred, rmses, mses, bp):
    if not os.path.exists(f"{directory}/dumps/{train_size}/{seed}"):
        os.makedirs(f"{directory}/dumps/{train_size}/{seed}")
        
    np.save(f"{directory}/dumps/{train_size}/{seed}/y_pred", y_pred) # noqa: W1514
    np.save(f"{directory}/dumps/{train_size}/{seed}/rmses", rmses) # noqa: W1514
    np.save(f"{directory}/dumps/{train_size}/{seed}/bpoints", bp) # noqa: W1514

    with open(f"{directory}/dumps/{train_size}/{seed}/mses.json", "w+", encoding='utf-8') as f:
        json.dump(
            dict(zip(["mse_treeningul", "mse_grid_testil", "mse_treening_andmete_teine_myra"],
                     map(lambda x: float(x.numpy()), mses))),
            f)
        

def samad_punktid_kui_treeningul_teine_myra(X_train, model, fn, noise_fn, test_goal):
    X_test = np.repeat(X_train, test_goal/len(X_train))
    y_test = fn(X_test) + noise_fn(X_test)
    y_pred = model.predict(X_test, batch_size=65536)
    return keras.losses.mean_squared_error(y_test, y_pred[:, 0])


def main():
    train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
               190, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10**4, 10**5, 10**6]
    X_grid_test = np.linspace(0, 10, 10**7)
    y_grid_test = fn(X_grid_test) + noise_fn(X_grid_test)
    test_goal = 10**7

    n_runs = 5
    seeds = np.arange(n_runs)
    epochs = 1000
    cur_dir = os.getcwd()
    directory = "const_func_uniform_data"
    directory = os.path.join(cur_dir, directory)
    print("saving to", directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(f"{directory}/dumps"):
        os.makedirs(f"{directory}/dumps")
    if not os.path.exists(f"{directory}/plots/"):
        os.makedirs(f"{directory}/plots/variance")
        os.makedirs(f"{directory}/plots/rmses")
    


    for train_size in train_sizes:    # kõik suurused
        print("Training size:", train_size)
        for seed in seeds:  # kõik seedid
            print("Training size:", train_size, "with seed:", seed)
            utils.reset_seeds(seed)
            X_train = np.random.uniform(0, 10, train_size)
            model, _h, (X_train, y_train) = train_model(
                X_train, nn_size=(20, 20), epochs=epochs,
                fn=fn, noise_fn=noise_fn, seed=seed
            )
            y_pred_train = model.predict(X_train, batch_size=65536, verbose=0)
            y_pred_grid = model.predict(
                X_grid_test, batch_size=65536, verbose=0)

            mse_treeningul = keras.losses.mean_squared_error(
                y_train, y_pred_train[:, 0])
            mse_grid_testil = keras.losses.mean_squared_error(
                y_grid_test, y_pred_grid[:, 0])
            mse_treening_andmete_teine_myra = samad_punktid_kui_treeningul_teine_myra(
                X_train, model, fn, noise_fn, test_goal)

            model.save(
                f"{directory}/models/{train_size}_{seed}", overwrite=True)
            # m = tf.keras.models.load_model("test2/19_hpc_script/models/10_0", custom_objects={'neg_log_likelihood': utils.neg_log_likelihood})

            variance_fig = utils.joonista_variance(model, X_test=np.linspace(-2, 12, 1000), X_train=X_train, y_train=y_train,
                                                   xlim=(-2, 12), ylim=(-10, 10), ground_truth=fn, bpoint_fn=Slopes.new_breakpoint_finder, return_fig=True)
            variance_fig.savefig(
                f"{directory}/plots/variance/{train_size}_{seed}.png")

            rmses, rmse_fig = utils.joonista_rmses5x(model=model, start=0, end=10, steps=1000, akna_laius=0.1,
                                                     fn=lambda x: 0*x, analyytiline_myra=lambda x: 0.09*x**2+0.09, show_plt=False)
            rmse_fig.savefig(
                f"{directory}/plots/rmses/{train_size}_{seed}.png")

            bpoints = Slopes.new_breakpoint_finder(model, np.linspace(-2, 12, 10**6))
            bpoints = [bp[0] for bp in bpoints]

            save_everything(directory, train_size, seed, y_pred=y_pred_grid, rmses=rmses, mses=[
                            mse_treeningul, mse_grid_testil, mse_treening_andmete_teine_myra], bp=bpoints)

        #     break
        # break


if __name__ == "__main__":
    main()
