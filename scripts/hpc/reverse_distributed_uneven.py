import sys
import os


print(os.getcwd())
# sys.path.append(os.path.dirname(os.getcwd())) # add parent directory to path
sys.path.append(os.getcwd())  # add  directory to path


import argparse
import scripts.Slopes as Slopes
import scripts.utils as utils
from scripts.CustomCallbacks import CustomLogger
import seaborn as sns
from matplotlib import pyplot as plt
from keras import layers
from tensorflow import keras
import tensorflow as tf
import random
import numpy as np
import json


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "-s", "--seed", help="random seed", type=int, default=0)
arg_parser.add_argument("-m", "--multiplier",
                        help="multiplier", type=str,)

arg_parser.add_argument(
    "-o", "--output", help="directory where to save the results", default="../hpc_results")

args = arg_parser.parse_args()

utils.reset_seeds(0)

print(args.seed)
print(args.multiplier)
print(args.output)

random.seed(args.seed)
print(f"{args.seed} sai sellise: {random.random()}")



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
    """Mudeli treenimine, voib kasutada erinevaid funktsioone ja mÃ¼ra, tagastab mudeli, ajaloo ja treeningandmed

    :param trainX: treeningX andmed
    :param nn_size: vorgu kihtide suurused, hetkel ainult 2 kihti lubatud, defaults to (20, 20)
    :param optimizer: optimiseerija, hetkel valitud hea vÃ¤iksematele andmehulkadele, defaults to keras.optimizers.Adam( learning_rate=0.003, amsgrad=True, epsilon=1e-3)
    :param epochs: epochide kogus, defaults to 1000
    :param batch_size: batch size, kasutab valemit min(16, kahe aste mis > len(X)/5), defaults to 16
    :param fn: funktsioon mida oppida, defaults to lambdax:0*x
    :param noise_fn: mÃ¼rafn, defaults to lambdax:0.3*np.random.randn(len(x))+0.3*x*np.random.randn(len(x))
    :param reverse_fn: kas pÃ¶Ã¶rata fn tagurpidi, defaults to False
    :param reverse_noise: kas pÃ¶Ã¶rata mÃ¼ra tagurpidi, defaults to False
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

    # koik voimalikud kombod reverse_fn ja reverse_noise. Default on tavalist pidi
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


def save_everything(directory, train_size, seed, different_multiplier, different_place, y_pred, rmses, mses, bp):

    if not os.path.exists(f"{directory}/dumps/{train_size}/{seed}/{different_multiplier}{chr(65+different_place)}"):
        os.makedirs(
            f"{directory}/dumps/{train_size}/{seed}/{different_multiplier}{chr(65+different_place)}")

    np.save(f"{directory}/dumps/{train_size}/{seed}/{different_multiplier}{chr(65+different_place)}/y_pred", y_pred)  # noqa: W1514
    np.save(f"{directory}/dumps/{train_size}/{seed}/{different_multiplier}{chr(65+different_place)}/rmses", rmses)  # noqa: W1514
    np.save(f"{directory}/dumps/{train_size}/{seed}/{different_multiplier}{chr(65+different_place)}/bpoints", bp)  # noqa: W1514

    with open(f"{directory}/dumps/{train_size}/{seed}/{different_multiplier}{chr(65+different_place)}/mses.json", "w+", encoding='utf-8') as f:
        json.dump(
            dict(zip(["mse_treeningul", "mse_grid_testil", "mse_treening_andmete_teine_myra"],
                     map(lambda x: float(x.numpy()), mses))),
            f)


def samad_punktid_kui_treeningul_teine_myra(X_train, model, fn, noise_fn, test_goal):
    X_test = np.repeat(X_train, test_goal/len(X_train))
    y_test = fn(X_test) + noise_fn(X_test)
    y_pred = model.predict(X_test, batch_size=65536)
    return keras.losses.mean_squared_error(y_test, y_pred[:, 0])


# seda tahame itereerida Ã¼le
# train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 750, 1000]
# different_place = [0, 1, 2, 3, 4]
# different_multiplier = [0.25, 0.5, 1, 2, 4]
# seed = [0, 1, 2, 3, 4]
def create_x_train(train_size, different_place, different_multiplier=1, x_range=(0, 10), n_places=5, seed=None):
    utils.reset_seeds(seed)

    keskmine_num = train_size/n_places  # A
    erinev_num = different_multiplier*keskmine_num  # B
    piirkond = (x_range[1]-x_range[0])/n_places
    piirkonnad = [(x_range[0]+i*piirkond, x_range[0]+(i+1)*piirkond)
                  for i in range(n_places)]
    #print(f"piirkond: {piirkond}")
    #print(f"piirkonnad: {piirkonnad}")

    X_train = []
    for idx, p in enumerate(piirkonnad):
        if idx == different_place:
            X_train.append(np.random.uniform(p[0], p[1], int(erinev_num)))
        else:
            X_train.append(np.random.uniform(p[0], p[1], int(keskmine_num)))

    X_train = np.concatenate(X_train)
    return X_train


def main():
    global args

    train_sizes = [100, 500, 2000] #[10, 20, 30, 40, 50, 60, 70, 80, 90,100, 125, 150, 175, 200, 250, 500, 750, 1000, 2000]
    different_places = [0,2,4] #[0, 1, 2, 3, 4]
    different_multiplier = float(args.multiplier)
    directory = args.output

    X_grid_test = np.linspace(0, 10, 10**6)
    y_grid_test = fn(X_grid_test) + noise_fn(X_grid_test)
    test_goal = 10**6

    seed = int(args.seed)
    epochs = 1000
    cur_dir = os.getcwd()


    directory = os.path.join(cur_dir, directory)
    print("saving to", directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(f"{directory}/dumps"):
        os.makedirs(f"{directory}/dumps")
    if not os.path.exists(f"{directory}/plots/"):
        os.makedirs(f"{directory}/plots/variance")
        os.makedirs(f"{directory}/plots/rmses")

    for train_size in train_sizes:    # koik suurused
        print("Training size:", train_size)
        for different_place in different_places:  # koik erinevad kohad
            print("Different place:", chr(65+different_place))
            print("Different multiplier:", different_multiplier)

            descriptor = f"train_size:{train_size}, diff:{different_multiplier}{chr(65+different_place)}, seed:{seed}"

            print(f"Training size: {train_size}, different place: {chr(65+different_place)}, different multiplier: {different_multiplier}, seed: {seed}")

            utils.reset_seeds(seed)
            X_train = create_x_train(train_size, different_place=different_place,
                                        different_multiplier=different_multiplier, x_range=(0, 10), n_places=5, seed=seed)

            model, _h, (X_train, y_train) = train_model(
                X_train, nn_size=(20, 20), epochs=epochs,
                fn=fn, noise_fn=noise_fn, seed=0,    #TODO important: kaalud algvaartustatakse alati sama seediga
                reverse_noise=True,
            )
            y_pred_train = model.predict(
                X_train, batch_size=65536, verbose=0)
            y_pred_grid = model.predict(
                X_grid_test, batch_size=65536, verbose=0)

            mse_treeningul = keras.losses.mean_squared_error(
                y_train, y_pred_train[:, 0])
            mse_grid_testil = keras.losses.mean_squared_error(
                y_grid_test, y_pred_grid[:, 0])
            mse_treening_andmete_teine_myra = samad_punktid_kui_treeningul_teine_myra(
                X_train, model, fn, noise_fn, test_goal)

            model.save(
                f"{directory}/models/{train_size}/{seed}/{different_multiplier}{chr(65+different_place)}", overwrite=True)
            # m = tf.keras.models.load_model("test2/19_hpc_script/models/10_0", custom_objects={'neg_log_likelihood': utils.neg_log_likelihood})

            variance_fig = utils.joonista_variance(model, X_test=np.linspace(-2, 12, 1000), X_train=X_train, y_train=y_train,
                                                    xlim=(-2, 12), ylim=(-10, 10), ground_truth=fn, bpoint_fn=Slopes.new_breakpoint_finder, return_fig=True, title_text=descriptor)
            variance_fig.savefig(
                f"{directory}/plots/variance/{train_size}_{different_multiplier}{chr(65+different_place)}_{seed}.png")

            rmses, rmse_fig = utils.joonista_rmses5x(model=model, start=0, end=10, steps=1000, akna_laius=0.1,
                                                        fn=lambda x: 0*x, analyytiline_myra=lambda x: 0.09*x**2+0.09, reverse=True, show_plt=False, title_text=descriptor)
            rmse_fig.savefig(
                f"{directory}/plots/rmses/{train_size}_{different_multiplier}{chr(65+different_place)}_{seed}.png")

            bpoints = Slopes.new_breakpoint_finder(
                model, np.linspace(-2, 12, 10**6))
            bpoints = [bp[0] for bp in bpoints]

            save_everything(directory, train_size, seed, different_multiplier, different_place, y_pred=y_pred_grid, rmses=rmses, mses=[
                mse_treeningul, mse_grid_testil, mse_treening_andmete_teine_myra], bp=bpoints)


if __name__ == "__main__":
    main()
