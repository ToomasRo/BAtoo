import sys
import os

print(os.getcwd())
sys.path.append(os.getcwd())  # add  directory to path

import scripts.utils as utils
import pickle
import json
import pandas as pd
import numpy as np
import argparse


utils.reset_seeds(0)

regioon = ['a', 'b', 'c', 'd', 'e']
naabrid = [['b'], ['a', 'c'], ['b', 'd'], ['c', 'e'], ['d']]
kaugemad = [['c'], ['d'], ['a', 'e'], ['b'], ['c']]

regioonide_combod = list(zip(regioon, naabrid, kaugemad))

with open('ols_model.pkl', 'rb') as f:
    regressioon_mudel = pickle.load(f)
with open('regression_scaler.pkl', 'rb') as f:
    reg_scaler = pickle.load(f)


def process_break_points(file_path, regions):
    with open(file_path, 'rb') as f:
        data = np.load(f)
    # print(data)
    return np.histogram(data[:, 0], bins=regions)[0]


def process_rmse(file_path, regions):
    with open(file_path, 'rb') as f:
        data = np.load(f)

    # splitime andmed vastavalt regionitele jätame välja piirkonnad mis on väiksem ja suurem kui andmete piirid
    r_all = np.split(data, np.searchsorted(data[:, 0], v=regions))[1:-1]

    # absoluutsed vahed joonte 2. ja 3. vahel (keskmine piirkonnas)
    absoluut_vahed = [np.mean(r[:, 2] - r[:, 3]) for r in r_all]

    # 3. joone keskmine igas piirkonnas
    r3_mean = [np.mean(r[:, 3]) for r in r_all]
    # 2. joone keskmine igas piirkonnas
    r2_mean = [np.mean(r[:, 2]) for r in r_all]

    # suhtelised vahed joonte 2. ja 3. vahel (keskmine piirkonnas)
    suhtelised_vahed = [absoluut / r_m for absoluut,
                        r_m in zip(absoluut_vahed, r3_mean)]

    return absoluut_vahed, suhtelised_vahed, (r2_mean, r3_mean)


def laisk_nll(y_true, ypredmean, ypredlogvar):
    # tf.reduce_mean(y_pred_var + tf.math.square(y_true - y_pred_mean) / tf.math.exp(y_pred_var))
    return np.mean(ypredmean + (y_true - ypredmean)**2 / np.exp(ypredlogvar))


def kai_labi_dir(baasdir):
    df_columns = ['train_size', 'random_seed', 'multiplier', 'region',
                  *[f'bpoints_in_{chr(a)}' for a in range(97, 97+5)],
                  *[f'rel_diff_in_{chr(a)}' for a in range(97, 97+5)],
                  *[f'raw_mean2_in_{chr(a)}' for a in range(97, 97+5)],
                  'mse_treeningul', ]

    uus_df_temp = []
    uus_df_cols = ['train_size', 'seed',
                   'multiplier', 'region', 'vana_nll', 'uus_nll']

    for dirpath, dirs_in_dir, files in os.walk(baasdir):
        if "models" in dirpath:
            continue
        if "plots" in dirpath:
            continue
        if len(files) != 0:
            # print(dirpath)
            head, multiplier_region = os.path.split(dirpath)
            head, seed = os.path.split(head)
            head, train_size = os.path.split(head)
            multiplier, region = float(
                multiplier_region[:-1]), multiplier_region[-1]
            seed, train_size = int(seed), int(train_size)

            if "sin" in dirpath:
                def fn(x): return x*np.sin(x)
            else:
                def fn(x): return 0*x

            def noise_fn(X): return 0.3 * np.random.randn(len(X)
                                                          ) + 0.3 * X * np.random.randn(len(X))
            if "reverse" in dirpath:
                reverse_noise = True
            else:
                reverse_noise = False

            bpoints = process_break_points(os.path.join(
                dirpath, "bpoints.npy"), np.arange(0, 12, 2))
            abs_diff, rel_diff, (raw_mean2, raw_mean3) = process_rmse(
                os.path.join(dirpath, "rmses.npy"), np.arange(0, 12, 2))
            with open(os.path.join(dirpath, "mses.json"), 'r', encoding="cp1252") as f:
                mses = json.load(f)
                mse_treeningul = mses["mse_treeningul"]

            with open(os.path.join(dirpath, "y_pred.npy"), 'rb') as f:
                y_preds = np.load(f)

            andmestik = pd.DataFrame(data=[[train_size, seed, multiplier, region,
                                     *bpoints, *rel_diff, *raw_mean2, mse_treeningul]], columns=df_columns)

            Xid, yid = [], []
            for r, n, k in regioonide_combod:

                X, y = utils.transform_no_scaling(andmestik, r, n, k)
                Xid.append(X)
                yid.append(y)

            suurX = pd.concat(Xid, ignore_index=True)
            suury = pd.concat(yid, ignore_index=True)
            # print("train_size, seed, multiplier,region")
            # print(train_size, seed, multiplier,region)
            # display(suurX.head())

            uuritavad_col = suurX.columns
            suurX = reg_scaler.transform(suurX,)
            suurX = pd.DataFrame(suurX, columns=uuritavad_col)
            suurX["my_points_combo"] = suurX.my_points * -0.3762 + suurX.my_points_m1 * -3.3876 + \
                suurX.my_points_log * 1.3885 + suurX.my_points_logm1 * \
                4.0044 + np.ones(shape=suurX.shape[0]) * -1.0568
            suurX["neighbour_points_combo"] = suurX.neighbour_points * -0.3268 + suurX.neighbour_points_m1 * -1.7966 + \
                suurX.neighbour_points_log * 0.7729 + suurX.neighbour_points_logm1 * \
                1.7054 + np.ones(shape=suurX.shape[0]) * -0.5441
            # suurX["distant_points_combo"] = suurX.distant_points * result3.params["distant_points"] + suurX.distant_points_m1 * result3.params["distant_points_m1"] + suurX.distant_points_log * result3.params["distant_points_log"] + suurX.distant_points_logm1 * result3.params["distant_points_logm1"] + np.ones(shape=suurX.shape[0]) * result3.params["const"]
            suurX["points_combo"] = suurX.my_points_combo * 0.8223 + \
                suurX.neighbour_points_combo * \
                0.2080  # + const4 * result4.params["const"]

            suurX["bpoints_combo"] = suurX.my_bpoints * -0.4097 + suurX.neighbour_bpoints * - \
                0.1139 + suurX.distant_bpoints * -0.0604 + \
                np.ones(shape=suurX.shape[0]) * 0.0618

            suurX.drop(['my_bpoints', 'neighbour_bpoints', 'distant_bpoints',
                        'my_points', 'neighbour_points_combo', 'my_points_combo',
                        'neighbour_points', 'distant_points', 'my_points_m1', 'my_points_log',
                        'my_points_logm1', 'neighbour_points_m1', 'neighbour_points_log',
                        'neighbour_points_logm1', 'distant_points_m1', 'distant_points_log',
                        'distant_points_logm1', ], axis=1, inplace=True)

            y_pred_mean = y_preds[:, 0]
            y_pred_logvar = y_preds[:, 1]
            a = y_pred_logvar
            y_pred_sd = np.sqrt(np.exp(y_pred_logvar))
            correctors = regressioon_mudel.predict(suurX)
            b = correctors.repeat(y_preds.shape[0]/5)

            x = a / (b+1)  # ühik : logvar
            x_sd = np.sqrt(np.exp(x))

            utils.reset_seeds(seed)

            X = utils.create_x_train(train_size=train_size, different_place=ord(
                region)-65, different_multiplier=multiplier)
            X = np.linspace(0, 10, 1000000)
            if reverse_noise:
                y_true = fn(X) + noise_fn(X[::-1])
            else:
                y_true = fn(X) + noise_fn(X)
            vana = laisk_nll(y_true, y_pred_mean, y_pred_logvar)
            uus = laisk_nll(y_true, y_pred_mean, x_sd)
            uus_df_temp.append(
                [train_size, seed, multiplier, region, vana, uus])

    return pd.DataFrame(data=uus_df_temp, columns=uus_df_cols)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-i", "--input",
                            help="input folder where all the results are located", type=str,)
    arg_parser.add_argument(
        "-o", "--output", help="directory where to save the results", default="../hpc_results")
    arg_parser.add_argument(
        "-f", "--file", help="file to save the results", default="out.csv")

    args = arg_parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    output_file = args.file

    df = kai_labi_dir(input_dir)

    print("Saving dataframe")
    df.to_csv(os.path.join(output_dir, output_file), index=False)