import os
from os.path import join as oj
from collections import defaultdict

import numpy as np
np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
if find_executable('latex'): 
    print('latex installed')
    plt.rcParams['text.usetex'] = True

import math
import random

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

import random

from collections import defaultdict

import torch
import gpytorch
from scipy.stats import dirichlet

from utils import Hellinger_dist, Chernoff_dist


import pandas as pd
def get_observed_expected_statistics(alphas, count=5000):

    h_bars = []
    for alpha in alphas:
        samples = dirichlet.rvs(alpha, size=count, random_state=42)
        # alpha_hat = mle(samples)
        # samples = dirichlet.rvs(alpha_hat, size=count, random_state=42)        
        samples = samples.clip(min =1e-10)
        # samples[samples == 0 ] = 1e-10

        mean = np.mean(np.log(samples), axis=0)
        h_bars.append(mean)

    return np.asarray(h_bars)

def load_SV_data(SV_data_dir, data_dir):
    # iN = int(data_dir.split('_')[0])
    # num_cluster = int(data_dir.split('_')[0].split('-')[1])
    # cluster_size = int(data_dir.split('-')[-1])

    SVs_C = np.loadtxt(oj(SV_data_dir, data_dir, 'SVs_C'))
    SVs_H = np.loadtxt(oj(SV_data_dir, data_dir, 'SVs_H'))
    alphas = np.loadtxt(oj(SV_data_dir, data_dir, 'alphas'))

    num_cluster = 3
    cluster_size = len(SVs_H) // 3

    # using h_bars as the features directly
    h_bars = get_observed_expected_statistics(alphas)
    print("Num Clusters : {}, cluster size: {}, alpha dimension: {}".format(num_cluster, cluster_size, np.shape(alphas)))

    return num_cluster, cluster_size, SVs_C, SVs_H, alphas, h_bars


from utils_GP import train, predict, SEHellingerKernel, SEChernoffKernel, CustomGPModel, ExactGPModel
import json

def save_dicts(to_save_dir, MSE_perfs, MaxE_perfs, r2_perfs, exVar_perfs):
    
    # dicts_dir = oj(to_save_dir)    
    os.makedirs(to_save_dir, exist_ok = True)

    for i in range(len(MSE_perfs)):

        specific_dir = oj(to_save_dir, str(i))
        os.makedirs(specific_dir, exist_ok = True)
        with open(oj(specific_dir, 'MSE_perf.json'), 'w') as f:
            json.dump(MSE_perfs[i], f)

        with open(oj(specific_dir, 'MaxE_perf.json'), 'w') as f:
            json.dump(MaxE_perfs[i], f)

        with open(oj(specific_dir, 'r2_perf.json'), 'w') as f:
            json.dump(r2_perfs[i], f)

        with open(oj(specific_dir, 'exVar_perf.json'), 'w') as f:
            json.dump(exVar_perfs[i], f)

    return

def get_train_test_kernels(over_all_kernel, train_indices, test_indices):

    N, M = len(train_indices), len(test_indices)
    train_kernel = np.zeros((N , N))
    for i, train_i in enumerate(train_indices):     
        for j, train_j in enumerate(train_indices):
            train_kernel[i,j] = over_all_kernel[train_i, train_j]

    test_kernel = np.zeros((M, N))
    for j, train_j in enumerate(test_indices):
        for i, train_i in enumerate(train_indices):     
            test_kernel[j, i] = over_all_kernel[train_i, train_j]

    return train_kernel, test_kernel


def precompute_kernels(alphas, lengscale=1):
    N = len(alphas)
    RBF_K_Hellinger = np.zeros((N, N))
    RBF_K_Chernoff = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            dist_H = Hellinger_dist(alphas[i], alphas[j])
            dist_C = Chernoff_dist(alphas[i], alphas[j])

            RBF_K_Hellinger[i][j] = math.exp( - dist_H**2 / (2*lengscale**2) )
            RBF_K_Chernoff[i][j] = math.exp( - dist_C**2 / (2*lengscale**2) )

    return RBF_K_Hellinger, RBF_K_Chernoff

def GPRegression(train_ratio=0.3):

    MSE_perfs = [defaultdict(dict) for  i in range(4) ]
    MaxE_perfs = [defaultdict(dict) for  i in range(4) ]
    r2_perfs = [defaultdict(dict) for  i in range(4) ]
    exVar_perfs = [defaultdict(dict) for  i in range(4) ]

    MSE_values = [0 for i in range(4)]
    MaxE_values = [0 for i in range(4)]
    r2_values = [0 for i in range(4)]
    exVar_values = [0 for i in range(4)]

    def update_performance_dict(Y_test, Y_pred, index):

        mse = MSE(Y_test, Y_pred)
        max_e = max_error(Y_test, Y_pred)
        exVar = explained_variance_score(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        MSE_perfs[index][num_cluster][cluster_size] = mse
        MaxE_perfs[index][num_cluster][cluster_size] = max_e
        exVar_perfs[index][num_cluster][cluster_size] = exVar
        r2_perfs[index][num_cluster][cluster_size] = r2

        MSE_values[index] += mse
        MaxE_values[index] += max_e
        r2_values[index] += exVar
        exVar_values[index] += r2

        return

    count = 0
    print('GP RBF -----')
    for data_dir in os.listdir(SV_data_dir):
        if '_byclass' in data_dir or '.png' in data_dir : continue

        count += 1
        try:
            num_cluster, cluster_size, SVs_C, SVs_H, alphas, h_bars = load_SV_data(SV_data_dir, data_dir)
        except Exception as e:
            print("Skipping data dir due to past data saving format:", data_dir)
            print(str(e))
            continue
        N = len(alphas)

        indices = np.arange(N)
        random.shuffle(indices)
        pos = int(train_ratio * N)

        train_indices, test_indices = indices[: pos], indices[pos:]


        # ----------------- Hellinger vs. SVs_H

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        X_train = torch.from_numpy(alphas).float()[train_indices]
        X_test = torch.from_numpy(alphas).float()[test_indices]

        Y_train = torch.from_numpy(SVs_H).float()[train_indices]
        Y_test = SVs_H[test_indices]
        
        model = CustomGPModel(X_train, Y_train, likelihood, SEHellingerKernel())

        # set to training mode and train
        train(model, likelihood)

        observed_pred = predict(model, likelihood, X_test)
        Y_pred = observed_pred.mean.numpy()

        update_performance_dict(Y_test, Y_pred, 0)

        # ----------------- Chernoff vs. SVs_C
        X_train = torch.from_numpy(alphas).float()[train_indices]
        X_test = torch.from_numpy(alphas).float()[test_indices]

        Y_train = torch.from_numpy(SVs_C).float()[train_indices]
        Y_test = SVs_C[test_indices]
        
        model = CustomGPModel(X_train, Y_train, likelihood, SEChernoffKernel())

        # set to training mode and train
        train(model, likelihood)

        observed_pred = predict(model, likelihood, X_test)
        Y_pred = observed_pred.mean.numpy()

        update_performance_dict(Y_test, Y_pred, 1)


        # ----------------- h_bars vs. SVs_H

        X_train = torch.from_numpy(h_bars).float()[train_indices]
        X_test = torch.from_numpy(h_bars).float()[test_indices]

        Y_train = torch.from_numpy(SVs_H).float()[train_indices]
        Y_test = SVs_H[test_indices]
        
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train, Y_train, likelihood, args.l1)

        # set to training mode and train
        train(model, likelihood)

        observed_pred = predict(model, likelihood, X_test)
        Y_pred = observed_pred.mean.numpy()
        update_performance_dict(Y_test, Y_pred, 2)


        # ----------------- h_bars vs. SVs_C

        X_train = torch.from_numpy(h_bars).float()[train_indices]
        X_test = torch.from_numpy(h_bars).float()[test_indices]

        Y_train = torch.from_numpy(SVs_C).float()[train_indices]
        Y_test = SVs_C[test_indices]

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train, Y_train, likelihood, args.l1)

        # set to training mode and train
        train(model, likelihood)

        observed_pred = predict(model, likelihood, X_test)
        Y_pred = observed_pred.mean.numpy()
        update_performance_dict(Y_test, Y_pred, 3)

    # to_save_dir = oj(save_results_dir,'GP-RBF', str(train_ratio))

    # save_dicts(to_save_dir, MSE_perfs, MaxE_perfs, r2_perfs, exVar_perfs)

    MSE_values = np.asarray(MSE_values) / count
    MaxE_values = np.asarray(MaxE_values) / count
    r2_values = np.asarray(r2_values) / count
    exVar_values = np.asarray(exVar_values) / count
    # df = pd.DataFrame(data = {'MSE':MSE_values, 'Max_E':MaxE_values, 'R2': r2_values, 'exVar': exVar_values}, index =['H','C','hbar-H','hbar-C'])

    # df.to_csv(oj(to_save_dir, 'overall.csv'))

    return train_ratio, [MSE_perfs, MaxE_perfs, r2_perfs, exVar_perfs], [MSE_values, MaxE_values, r2_values, exVar_values]


from multiprocessing import Pool

from scipy.stats import sem
import argparse


import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

if  __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process which dataset to perform learning of MSVs.')
    parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, default='MNIST')

    parser.add_argument('--l1', dest='l1', action='store_true')
    parser.add_argument('--not_l1', dest='l1', action='store_false') # default of using not_l1 is fine

    args = parser.parse_args()
    print(args)

    # NOTE: Need to provide the right directory
    SV_data_dir = '{}_results/precision'.format(args.dataset)
    # SV_data_dir = '{}_results'.format(args.dataset)

    if args.l1:
        save_results_dir = 'learn_{}-L1'.format(args.dataset)
    else:
        save_results_dir = 'learn_{}-L2'.format(args.dataset)

    LABEL_FONTSIZE = 20
    MARKER_SIZE = 10
    AXIS_FONTSIZE = 26
    TITLE_FONTSIZE= 26
    LINEWIDTH = 6

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('figure', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXIS_FONTSIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LABEL_FONTSIZE)    # legend fontsize
    plt.rc('lines', markersize=MARKER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=LINEWIDTH)  # fontsize of the figure title


    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    np.random.seed(42)
    random.seed(42)

    ress = []
    Repeat = 10
    train_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    with Pool(processes=8) as pool:

        for train_ratio in train_ratios:
            for repeat in range(Repeat):
                res = pool.apply_async(GPRegression, ([train_ratio]))      # runs in *only* one process
                ress.append(res)

        results = [result.get() for result in ress]

    perfs_by_ratio = defaultdict(list)
    values_by_ratio = defaultdict(list)

    average_perfs =  [defaultdict(dict) for  i in range(4) ]
    standard_error_perfs =  [defaultdict(dict) for  i in range(4) ]
    avg_stderr_perfs =  [defaultdict(dict) for  i in range(4) ]

    for (ratio, perfs, values) in results:
        # perfs_by_ratio[ratio].append(perfs)
        values_by_ratio[ratio].append(values)

    avg_overall, stderr_overall = [], []

    for ratio in train_ratios:
        values_repeat = values_by_ratio[ratio]
        
        values_repeat = np.asarray(values_repeat)

        avg_all = values_repeat.mean(axis=0)

        stderr_all = sem(values_repeat, axis=0)

        names =['MSE','MaxE','R2', 'exVar']
        
        data = {}
        for name, avg, stderr in zip(names, avg_all, stderr_all):
            data[name + '_mean'] = avg
            data[name + '_stderr'] = stderr
        df = pd.DataFrame(data, index =['H','C','hbar-H','hbar-C'] )


        to_save_dir = oj(save_results_dir, 'GP-RBF', str(ratio) + 'repeat'+str(Repeat) )
        os.makedirs(to_save_dir, exist_ok=True)
        df.to_csv(oj(to_save_dir, 'overall.csv'))

        avg_overall.append(avg_all)
        stderr_overall.append(stderr_all)

    avg_overall = np.asarray(avg_overall)
    stderr_overall = np.asarray(stderr_overall)

    np.savetxt(oj(save_results_dir, 'GP-RBF', 'MSE_mean.txt'), avg_overall[:, 0])
    np.savetxt(oj(save_results_dir, 'GP-RBF', 'MSE_sem.txt'),stderr_overall[:, 0])
    np.savetxt(oj(save_results_dir, 'GP-RBF', 'MaxE_mean.txt'), avg_overall[:, 1])
    np.savetxt(oj(save_results_dir, 'GP-RBF', 'MaxE_sem.txt'), stderr_overall[:, 1])
    np.savetxt(oj(save_results_dir, 'GP-RBF', 'R2_mean.txt'), avg_overall[:, 2])
    np.savetxt(oj(save_results_dir, 'GP-RBF', 'R2_sem.txt'), stderr_overall[:, 2])
    np.savetxt(oj(save_results_dir, 'GP-RBF', 'exVar_sem.txt'), avg_overall[:, 3])
    np.savetxt(oj(save_results_dir, 'GP-RBF', 'exVar_mean.txt'), stderr_overall[:, 3])

    fig, ax1 = plt.subplots(figsize=(6, 4))
        
    # MSE
    mean = avg_overall[:, 0, 1]
    std = stderr_overall[:, 0, 1]

    # ax1.plot(train_ratios, MSE, label=r'$\boldsymbol{\alpha}_i$', linestyle='--')
    ax1.errorbar(train_ratios, mean, std, label=r'$\boldsymbol{\alpha}_i$', fmt='--o', color='C0')

    # MSE hbar
    mean = avg_overall[:, 0, 3]
    std = stderr_overall[:, 0, 3]
    
    # ax1.plot(train_ratios, mean, label= r'$\bar{h}_i$', linestyle='--')
    ax1.errorbar(train_ratios, mean, std, label= r'$\bar{h}_i$', fmt='--o',color='C1')

    ax1.set_xlabel('Training data ratio')
    ax1.set_ylabel('MSE')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # MaxE 
    mean = avg_overall[:, 1, 1]
    std = stderr_overall[:, 1, 1]
    # ax2.plot(train_ratios, mean, label=r'$\boldsymbol{\alpha}_i$', )
    ax1.errorbar(train_ratios, mean, std, label=r'$\boldsymbol{\alpha}_i$', fmt='-o', color='C0')

    # MaxE hbar
    mean = avg_overall[:, 1, 3]
    std = stderr_overall[:, 1, 3]

    # ax2.plot(train_ratios, mean, label=r'$\bar{h}_i$', )
    ax1.errorbar(train_ratios, mean, std, label=r'$\bar{h}_i$', fmt='-o', color='C1')

    # ax2.set_xlabel('Training data ratio')
    ax2.set_ylabel('Max error')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(oj(save_results_dir, 'GP-RBF', 'error_vs_train_ratio.png'))
    # plt.show()
    plt.clf()
    plt.close()


    fig, ax1 = plt.subplots(figsize=(6, 4))
        
    # R2
    mean = avg_overall[:, 2, 1]
    std = stderr_overall[:, 2, 1]

    # ax1.plot(train_ratios, MSE, label=r'$\boldsymbol{\alpha}_i$', linestyle='--')
    ax1.errorbar(train_ratios, mean, std, label=r'$\boldsymbol{\alpha}_i$', fmt='--o', color='C0')

    # R2 hbar
    mean = avg_overall[:, 2, 3]
    std = stderr_overall[:, 2, 3]
    
    # ax1.plot(train_ratios, mean, label= r'$\bar{h}_i$', linestyle='--')
    ax1.errorbar(train_ratios, mean, std, label= r'$\bar{h}_i$', fmt='--o',color='C1')

    ax1.set_xlabel('Training data ratio')
    ax1.set_ylabel('$R^2$')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # exVar
    mean = avg_overall[:, 3, 1]
    std = stderr_overall[:, 3, 1]
    # ax2.plot(train_ratios, mean, label=r'$\boldsymbol{\alpha}_i$', )
    ax1.errorbar(train_ratios, mean, std, label=r'$\boldsymbol{\alpha}_i$', fmt='-o', color='C0')

    # exVar hbar
    mean = avg_overall[:, 3, 3]
    std = stderr_overall[:, 3, 3]

    # ax2.plot(train_ratios, mean, label=r'$\bar{h}_i$', )
    ax1.errorbar(train_ratios, mean, std, label=r'$\bar{h}_i$', fmt='-o', color='C1')

    # ax2.set_xlabel('Training data ratio')
    ax2.set_ylabel('ExVar')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(oj(save_results_dir, 'GP-RBF', 'r2_vs_train_ratio.png'))

    # plt.show()
    plt.clf()
    plt.close()

