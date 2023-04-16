import sys
import os

print(os.getcwd())
sys.path.append(os.getcwd())  # add  directory to path


import argparse
import numpy as np
import pandas as pd
import json



def process_break_points(file_path, regions):
    with open(file_path, 'rb') as f:
        data = np.load(f)
    # print(data)
    return np.histogram(data[:, 0], bins=regions)[0]


def process_break_points_laiem(file_path, regions):
    with open(file_path, 'rb') as f:
        data = np.load(f)
    result = np.zeros(len(regions))
    for idx, reg in enumerate(regions):
        
        result[idx] = len(np.where((data[:, 0] >= reg[0])
                          & (data[:, 0] < reg[1]))[0])

    return result

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
    r2_mean =  [np.mean(r[:, 2]) for r in r_all]

    # suhtelised vahed joonte 2. ja 3. vahel (keskmine piirkonnas)
    suhtelised_vahed = [absoluut / r_m for absoluut, r_m in zip(absoluut_vahed, r3_mean)]

    return absoluut_vahed, suhtelised_vahed, (r2_mean, r3_mean)

def kogu_andmed_dfks(pathikene: str, df_columns: list[str]):
    root_dir = os.path.join(pathikene)
    plot_paths = []
    mse_paths = []
    regions = np.arange(0, 12, 2)

    _temp = []
    _cnt = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        hist_bpoint = None
        hist_bpoint_laiem = None
        abs_diff = None
        rel_diff = None
        mses = None
        print(dirpath)

        for filename in filenames:
            # print(os.path.join(dirpath, filename))
            # print(filenames)

            #print(f"train size: {train_size}, seed: {seed}, multiplier_region: {multiplier}, region {region}")

            if filename.endswith('.npy'):
                
                _cnt += 1
                head, multiplier_region = os.path.split(dirpath)
                head, seed = os.path.split(head)
                head, train_size = os.path.split(head)
                multiplier, region = float(multiplier_region[:-1]), multiplier_region[-1]
                if "bpoints" in filename:
                    hist_bpoint = process_break_points(os.path.join(dirpath, filename), regions)
                    hist_bpoint_laiem = process_break_points_laiem(os.path.join(dirpath, filename), [(-1, 3), (1, 5), (3, 7), (5, 9), (7, 11)])
                elif "rmses" in filename:
                    abs_diff, rel_diff, (raw_mean2, raw_mean3) = process_rmse(os.path.join(dirpath, filename), regions)
                elif "y_pred" in filename:
                    ...
            if filename.endswith('.json'):
                with open(os.path.join(dirpath, filename), 'r', encoding="cp1252") as f:
                    mses = json.load(f)
                
                

        if hist_bpoint is not None and abs_diff is not None and rel_diff is not None and hist_bpoint_laiem is not None and mses is not None:
            _temp.append([train_size, seed, multiplier, region, *hist_bpoint, *hist_bpoint_laiem, *abs_diff, *rel_diff, *raw_mean2, *raw_mean3, *mses.values()])
    
    print("Creating dataframe")
    df = pd.DataFrame(_temp, columns=df_columns)
    return df

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-i", "--input",
                            help="input folder where all the results are located", type=str,)
    arg_parser.add_argument(
        "-o", "--output", help="directory where to save the results", default="../hpc_results")
    arg_parser.add_argument("-f", "--file", help="file to save the results", default="out.csv")

    args = arg_parser.parse_args()

    df_columns = ['train_size', 'random_seed', 'multiplier', 'region',
              *[f'bpoints_in_{chr(a)}' for a in range(97, 97+5)],
              *[f'ext_bpoints_in_{chr(a)}' for a in range(97, 97+5)],
              *[f'abs_diff_in_{chr(a)}' for a in range(97, 97+5)],
              *[f'rel_diff_in_{chr(a)}' for a in range(97, 97+5)],
              *[f'raw_mean2_in_{chr(a)}' for a in range(97, 97+5)],
              *[f'raw_mean3_in_{chr(a)}' for a in range(97, 97+5)],
              *['mse_treeningul', 'mse_grid_testil', 'mse_treening_andmete_teine_myra'],]

    input_dir = args.input
    output_dir = args.output
    output_file = args.file

    df = kogu_andmed_dfks(input_dir, df_columns)
    print("Saving dataframe")
    df.to_csv(os.path.join(args.output, args.file), index=False)