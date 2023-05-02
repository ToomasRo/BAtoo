import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
 
from scripts import Slopes
import matplotlib.pyplot as plt
import seaborn as sns



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
    ax1.plot(X_pred, m.predict(X_pred, batch_size=32768,), label="prediction",  linewidth=0.5)
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


def joonista_variance(m: tf.keras.Sequential, X_test, X_train, y_train=None, bpoint_fn=Slopes.breakpoint_finder, *, ground_truth=False, xlim=None, ylim=None, return_fig=False, no_variance=False, n_variances=2, title_text=""):

    bpoints = bpoint_fn(m, X_test)

    _patterns = [bp[1] for bp in bpoints]
    bpoints = [bp[0] for bp in bpoints]

    start, end = xlim if xlim else (-1, 1)

    print(f"model contains {len(bpoints)} breaks")

    y_pred = m.predict(X_test, batch_size=32768)

    if no_variance:
        y_pred_mean = y_pred
    else:
        y_pred_mean, y_pred_logvar = y_pred[:, 0], y_pred[:, 1]
        y_pred_var = np.exp(y_pred_logvar)
        y_pred_sd = np.sqrt(y_pred_var)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.xlim(xlim)
    plt.ylim(ylim)

    # absoluutselt õige
    if ground_truth:
        plt.plot(np.linspace(start, end, 1000), ground_truth(np.linspace(
            start, end, 1000)), label="ground truth", alpha=0.5, color="orange")

    plt.plot(X_test, y_pred_mean, label="mean")
    if not no_variance:
        plt.fill_between(X_test, y_pred_mean - n_variances*y_pred_sd,
                         y_pred_mean + n_variances*y_pred_sd, alpha=0.2, label=f"{n_variances} standard hälvet")
    plt.scatter(X_train, y_train, marker='.', color="red", label="train")

    temp_bpoints = list(zip(*bpoints))
    bx, by = temp_bpoints[0], temp_bpoints[1]

    plt.scatter(bx, by, marker="o", color="green", label="breaks")
    plt.title(title_text)
    plt.legend()

    if return_fig:
        return fig

    plt.show()



def neg_log_likelihood(y_true, y_predicted):

    y_pred_mean, y_pred_var = y_predicted[:, 0:1], y_predicted[:, 1:]

    # y_pred_var ongi log variance
    # return tf.reduce_mean(tf.math.square(y_true - y_pred_var))
    # return tf.reduce_mean(y_pred_var + 1.7 / tf.math.exp(y_pred_var))
    return tf.reduce_mean(y_pred_var + tf.math.square(y_true - y_pred_mean) / tf.math.exp(y_pred_var))


def nll_gaussian(y_pred_mean, y_pred_sd, y_test):

    # element wise square
    # preserve the same shape as y_pred.shape
    square = tf.math.square(y_pred_mean - y_test)
    ms = tf.math.add(tf.math.divide(square, y_pred_sd), tf.math.log(y_pred_sd))
    # axis = -1 means that we take mean across the last dimension
    # the output keeps all but the last dimension
    ## ms = tf.reduce_mean(ms,axis=-1)
    # return scalar

    # nemad teevad square/sd + log(sd)

    # ehk nemad teevad square/var**0.5 + 0.5log(var)

    # meie teeme square/var + log(var)

    return tf.reduce_mean(ms)


def neg_log_likelihood2(ytrue, ypreds):
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5*tf.reduce_sum(tf.math.square((ytrue-mu) /
                             tf.math.exp(logsigma)), axis=1)
    sigma_trace = -tf.reduce_sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)

    log_likelihood = mse+sigma_trace+log2pi

    return tf.reduce_mean(-log_likelihood)


def train_data_maker(elements: list[tuple[float, float, float, float]], fn=lambda x: x**2, seed=0) -> tuple[np.array, np.array]:
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


####################
#
# RMSE arvutamised
#
####################
def tegelik_myra(x, fn=lambda x: 0.09*x**2+0.09, reverse=False, end=10):
    """Tegelik myra, mis on funktsioonist arvutatud

    :param x: x
    :param fn: analüütiliselt võetud sd, defaults to lambda x: np.sqrt(0.09*x**2+0.09)
    :return: tegelik myra
    """
    # tegelik_myra_lambda = lambda x: np.sqrt(0.09*x**2+0.09)
    #x = 10-x
    if reverse:     # TODO: see töötab vaid siis kui uurimispiirkond on 0..10, vist töötab nüüd
        x = end-x
    return np.sqrt(fn(x))  # noqa: C3001


def rmse_from_model_variance_estimate(model, start, end, n_points=1000):
    """Joon nr2, mudeli poolt ennustatud variansi summast arvutatud RMSE,
    täpsemalt variancide keskmise ruutjuur

    :param model: _description_
    :param start: _description_
    :param end: _description_
    :param n_points: _description_, defaults to 1000
    :return: _description_
    """
    X = np.linspace(start, end, n_points)
    y_pred = model.predict(X, batch_size=32768, verbose=0)
    y_pred_variance = np.exp(y_pred[:, 1:])
    #assert abs(np.sum(y_pred_variance)/n_points - np.mean(y_pred_variance)) < 1e-5
    return np.sqrt(np.sum(y_pred_variance)/n_points)


def rmse_from_modelmean_2_samples(model, X_test, y_test):
    """Joon nr3, mudeli poolt ennustatud keskväärtuse ja punktide vaheline kaugus

    :param model: _description_
    :param X_test: _description_
    :param y_test: _description_
    :return: _description_
    """
    y_pred = model.predict(X_test, batch_size=32768, verbose=0)
    y_pred_mean = y_pred[:, 0]

    return np.sqrt(np.mean(
        (y_pred_mean-y_test)**2
    ))


def rmse_from_diff_means(model, start, end, fn, n_points=1000,):
    """Joon nr4, mudeli poolt ennustatud variansi summast arvutatud RMSE

    :param model: _description_
    :param start: _description_
    :param end: _description_
    :param n_points: _description_, defaults to 1000
    :return: _description_
    """
    X = np.linspace(start, end, n_points)
    y_pred = model.predict(X, batch_size=32768, verbose=0)
    y_pred_mean = y_pred[:, 0]
    y_true_mean = fn(X)
    return np.sqrt(np.mean(
        (y_true_mean-y_pred_mean)**2
    ))


def rmse_from_gtruth_2_samples(X_test, y_test, fn):
    """Joon nr5, testandmepunktide kaugus tõelisest funktsioonist

    :param X_test: testhulk
    :param y_test: testhulgale vastavate X-idele varem genetud y väärtused, et müra oleks kõikides katsetes sama
    :param fn: funktsioon mida ennustame
    :return: rmse v5
    """
    y_true_mean = fn(X_test)
    return np.sqrt(np.mean(
        (y_true_mean-y_test)**2
    ))


####################
#
# RMSE plotting
#
####################

def calculate_rmses(model, start=0, end=10, steps=1000, akna_laius=0.1, 
                    fn=lambda x: x*np.sin(x), noise_fn=lambda x:0.3 * np.random.randn(len(x)) + 0.3 * x * np.random.randn(len(x)), 
                    analyytiline_myra=lambda x: 0.09*x**2+0.09, 
                    reverse_fn=False, reverse_noise=False):
    reset_seeds(100)
    X_test = np.linspace(start-2, end+2, 100_000) # TODO see on tõeline X_test, ei saa parameetriga ette anda.
    y_test = fn(X_test) + noise_fn(X_test)
    
    if reverse_fn and reverse_noise:
        y_test = fn(X_test[::-1]) + noise_fn(X_test[::-1])
    elif reverse_fn:
        y_test = fn(X_test[::-1]) + noise_fn(X_test)
    elif reverse_noise:
        y_test = fn(X_test) + noise_fn(X_test[::-1])
    else:
        y_test = fn(X_test) + noise_fn(X_test)
    
    y_pred = model.predict(X_test,batch_size=32768, verbose=0)
    y_pred_mean = y_pred[:, 0]
    y_pred_variance = np.exp(y_pred[:,1:])
    y_pred_sd = np.sqrt(y_pred_variance)

    y_true_mean = fn(X_test)

    uuringuruum, _ = np.linspace(start, end, steps, retstep=True)
    rmses2 = []
    for w_start in uuringuruum:
        
        w_end = w_start + akna_laius
        X_aknas = X_test[(X_test >= w_start) & (X_test < w_end)]
        y_aknas = y_test[(X_test >= w_start) & (X_test < w_end)]
        
        y_pred_aknas = y_pred[(X_test >= w_start) & (X_test < w_end)]
        y_pred_mean_aknas = y_pred_mean[(X_test >= w_start) & (X_test < w_end)]
        y_pred_variance_aknas = y_pred_variance[(X_test >= w_start) & (X_test < w_end)]
        y_pred_sd_aknas = y_pred_sd[(X_test >= w_start) & (X_test < w_end)]

        y_true_mean_aknas = y_true_mean[(X_test >= w_start) & (X_test < w_end)]

        rmse1 = tegelik_myra((w_start+w_end)/2, fn=analyytiline_myra, reverse=reverse_noise, end=end)
        rmse2 = np.sqrt(np.mean(y_pred_variance_aknas))
        rmse3 =  np.sqrt(np.mean(
            (y_pred_mean_aknas-y_aknas)**2
        ))
        rmse4 =  np.sqrt(np.mean(
            (y_true_mean_aknas-y_pred_mean_aknas)**2
        ))
        rmse5 =  np.sqrt(np.mean(
            (y_true_mean_aknas-y_aknas)**2
        ))
        rmses2.append([w_start, rmse1, rmse2, rmse3, rmse4, rmse5])
        #print(f"{w_start:.2f} {rmse1:.2f} {rmse2:.2f} {rmse3:.2f} {rmse4:.2f} {rmse5:.2f}")

    return np.array(rmses2).reshape(-1, 6)


def joonista_rmses5x(model, start=0, end=10, steps=1000, akna_laius=0.1, x_lim=None,y_lim=(0,30), 
                     fn=lambda x: x*np.sin(x), noise_fn=lambda x:0.3 * np.random.randn(len(x)) + 0.3 * x * np.random.randn(len(x)), 
                    analyytiline_myra=lambda x: 0.09*x**2+0.09, 
                    reverse_fn=False, reverse_noise=False, 
                    show_plt=True, title_text=""):
    
    rmses = calculate_rmses(model, start=start, end=end, steps=steps, akna_laius=akna_laius, fn=fn, noise_fn=noise_fn, analyytiline_myra=analyytiline_myra, reverse_fn=reverse_fn, reverse_noise=reverse_noise)
    sns.set_style('ticks')

    fig, ax = plt.subplots()

    c1, c2, c3, c4, c5 = sns.color_palette("tab10", 5)
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=rmses[:, 1]**2, color=c1, label='müra valem (1)')
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=rmses[:, 2]**2, color=c2, label='mudeli variance kaudu (2)') # mudel annab välja log-variance. Ehk tegelikult tuleks võtta hoopis selle ruutjuur - ei ole, see on juba sd sest me võtsime ruutjuure enne
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=rmses[:, 3]**2, color=c3, label='mudeli keskväärtus vs samples (3)')
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=rmses[:, 4]**2, color=c4, label='keskväärtuste erinevus (4)')
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=rmses[:, 5]**2, color=c5, label='tõeline keskväärtus vs samples (5)') # esimesena plotitud et oleks kõige rohkem tagaplaanil.
    ax.legend()
    ax.set_ylabel('variance')

    ax.set_ylim(y_lim)

    plt.title(title_text+f", w={akna_laius}")
    
    if show_plt:
        plt.show()

    return rmses, fig


def joonista_hypotees_2gt1(rmses, akna_laius=0.1, a_min=0, a_max=15):
    fig, ax = plt.subplots()
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=rmses[:, 1]**2, label='müra valemist (1)')
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=rmses[:, 2]**2, label='mudeli variance\'id (2)')
    ax.set_ylabel('variance')
    plt.show()


def joonista_hypotees_143(rmses, akna_laius=0.1, a_min=0, a_max=15):
    _, ax = plt.subplots()
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=np.clip(rmses[:, 1]**2+rmses[:, 4]**2, a_min=a_min, a_max=a_max) , label='1.+4.')
    sns.lineplot(x=rmses[:, 0]+akna_laius/2, y=np.clip(rmses[:, 3]**2, a_min=a_min, a_max=a_max), label='3.')
    ax.set_ylabel('variance')
    

def read_aggregated_data(path):
    df = pd.read_csv(path)

    df.train_size = df.train_size.astype(int)
    df.random_seed = df.random_seed.astype(int)
    df.multiplier = df.multiplier.astype(float)
    for i in range(5):
        df[f"ext_bpoints_in_{chr(i+ord('a'))}"] = df[f"ext_bpoints_in_{chr(i+ord('a'))}"].astype(int)

    # lisame mugavusväärtused
    df["points_in_equal_regions"] = (df["train_size"] / 5).astype(int)
    df["points_in_diff_region"] = (
        df["points_in_equal_regions"] * df["multiplier"]).astype(int)
    df["total_points"] = 4*df["points_in_equal_regions"] + \
        df["points_in_diff_region"]

    # tuunime välja väga kauged väärtused:
    cols_to_process = [*[f'abs_diff_in_{chr(a)}' for a in range(97, 97+5)],
                       *[f'rel_diff_in_{chr(a)}' for a in range(97, 97+5)],
                       *[f'raw_mean2_in_{chr(a)}' for a in range(97, 97+5)],
                       *[f'raw_mean3_in_{chr(a)}' for a in range(97, 97+5)],
                       *['mse_treeningul', 'mse_grid_testil',
                           'mse_treening_andmete_teine_myra'],
                       ]

    df[cols_to_process] = df.loc[:, cols_to_process][np.abs(
        stats.zscore(df.loc[:, cols_to_process])) < 3]
    df.dropna(inplace=True)

    df.sort_values(by=["train_size", "region", "multiplier",
                   "random_seed"], inplace=True, ignore_index=True)
    df = df.groupby(['region', 'multiplier', 'train_size']
                    ).mean().reset_index()

    df["combo_name"] = df["train_size"].astype(str) + "_" + df["region"] + "_" + df["multiplier"].astype(str)
    return df


def transform_no_scaling(df, minu: str, naabrid: list[str], kaugemad: list[str]):
    ma_olen_reg = minu
    # naabrid = ['a','c']
    naabrid_upper = [n.upper() for n in naabrid]
    # kaugemad = ['d','e']
    kaugemad_upper = [k.upper() for k in kaugemad]

    # y_col = "rel_diff_in_"+ma_olen_reg

    df_new = pd.DataFrame()
    df_new["my_bpoints"] = df["bpoints_in_"+ma_olen_reg]
    if len(naabrid) == 2:
        df_new["neighbour_bpoints"] = (
            df["bpoints_in_"+naabrid[0]] + df["bpoints_in_"+naabrid[1]])/2
    else:
        df_new["neighbour_bpoints"] = df["bpoints_in_"+naabrid[0]]

    if len(kaugemad) == 2:
        df_new["distant_bpoints"] = (
            df["bpoints_in_"+kaugemad[0]] + df["bpoints_in_"+kaugemad[1]])/2
    else:
        df_new["distant_bpoints"] = df["bpoints_in_"+kaugemad[0]]

    df_new["my_rel_diff"] = df["rel_diff_in_"+ma_olen_reg]

    df_new["my_raw_mean2"] = df["raw_mean2_in_"+ma_olen_reg]
    if len(naabrid) == 2:
        df_new["neighbour_raw_mean2"] = (
            df["raw_mean2_in_"+naabrid[0]] + df["raw_mean2_in_"+naabrid[1]])/2
    else:
        df_new["neighbour_raw_mean2"] = df["raw_mean2_in_"+naabrid[0]]

    if len(kaugemad) == 2:
        df_new["distant_raw_mean2"] = (
            df["raw_mean2_in_"+kaugemad[0]] + df["raw_mean2_in_"+kaugemad[1]])/2
    else:
        df_new["distant_raw_mean2"] = df["raw_mean2_in_"+kaugemad[0]]

    # minus olevad punktid
    df_new["my_points"] = (df["train_size"]/5 * df["multiplier"] * (df["region"] ==
                           ma_olen_reg.upper())) + (df["train_size"]/5 * (df["region"] != ma_olen_reg.upper()))
    df_new["my_points"] = df_new["my_points"].astype(int)

    if len(naabrid) == 2:
        df_new["neighbour_points"] = (df["train_size"]/5 * df["multiplier"] * (
            df["region"] == naabrid_upper[0])) + (df["train_size"]/5 * (df["region"] != naabrid_upper[0]))
        df_new["neighbour_points"] += (df["train_size"]/5 * df["multiplier"] * (
            df["region"] == naabrid_upper[1])) + (df["train_size"]/5 * (df["region"] != naabrid_upper[1]))
        df_new["neighbour_points"] = (df_new["neighbour_points"]/2).astype(int)
    else:
        df_new["neighbour_points"] = (df["train_size"]/5 * df["multiplier"] * (
            df["region"] == naabrid_upper[0])) + (df["train_size"]/5 * (df["region"] != naabrid_upper[0]))
        df_new["neighbour_points"] = df_new["neighbour_points"].astype(int)

    if len(kaugemad) == 2:
        df_new["distant_points"] = (df["train_size"]/5 * df["multiplier"] * (
            df["region"] == kaugemad_upper[0])) + (df["train_size"]/5 * (df["region"] != kaugemad_upper[0]))
        df_new["distant_points"] += (df["train_size"]/5 * df["multiplier"] * (
            df["region"] == kaugemad_upper[1])) + (df["train_size"]/5 * (df["region"] != kaugemad_upper[1]))
        df_new["distant_points"] = (df_new["distant_points"]/2).astype(int)
    else:
        df_new["distant_points"] = (df["train_size"]/5 * df["multiplier"] * (
            df["region"] == kaugemad_upper[0])) + (df["train_size"]/5 * (df["region"] != kaugemad_upper[0]))
        df_new["distant_points"] = df_new["distant_points"].astype(int)

    # et ei oleks ühtegi nulli enam
    df_new["my_points"] = (df_new["my_points"]+1).astype(float)
    df_new["neighbour_points"] = (df_new["neighbour_points"]+1).astype(float)
    df_new["distant_points"] = (df_new["distant_points"]+1).astype(float)

    df_new["my_points_m1"] = df_new["my_points"] ** (-1)
    df_new["my_points_log"] = np.log(df_new["my_points"])
    df_new["my_points_logm1"] = (np.log(df_new["my_points"])+1.0) ** (-1)

    df_new["neighbour_points_m1"] = df_new["neighbour_points"] ** (-1)
    df_new["neighbour_points_log"] = np.log(df_new["neighbour_points"])
    df_new["neighbour_points_logm1"] = (
        np.log(df_new["neighbour_points"]) + 1.0) ** (-1)

    df_new["distant_points_m1"] = df_new["distant_points"] ** (-1)
    df_new["distant_points_log"] = np.log(df_new["distant_points"])
    df_new["distant_points_logm1"] = (
        np.log(df_new["distant_points"]) + 1.0) ** (-1)

    df_new["mse_treeningul"] = df["mse_treeningul"]
    df_new.dropna(inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    df_new.drop_duplicates(inplace=True, ignore_index=True)

    X = df_new.drop(columns=['my_rel_diff'])
    y = df_new['my_rel_diff']

    return X, y

def create_x_train(train_size, different_place, different_multiplier=1, x_range=(0, 10), n_places=5, seed=None):
    # utils.reset_seeds(seed)

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