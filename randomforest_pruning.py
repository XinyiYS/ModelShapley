from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer,fetch_covtype
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


import os
from os.path import join as oj
import numpy as np

from dirichlet import mle
from get_MSVs import get_SV_permu_count

def get_alpha(estimator, X, y):
    logits = estimator.predict_proba(X)
    logits = np.clip(logits, a_min=1e-3, a_max=1)
    logits = normed_matrix = normalize(logits, axis=1, norm='l1')
    alphas = mle(logits, method='fixpoint')
    return alphas

from multiprocessing import Pool
import multiprocessing

from copy import deepcopy

import matplotlib.pyplot as plt
from utils import cwd

from scipy.stats import sem

import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process which dataset to construct a deep ensemble for pruning.')
    parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, default='BreastCancer')
    parser.add_argument('-N', '--N', help='The number of base estimators', type=int, default=5)
    parser.add_argument('-M', '--num_trials', help='The number of random trials', type=int, default=10)
    parser.add_argument('-depth', '--max_depth', help='The maximum depth of the base tree estimators', type=int, default=3)

    parser.add_argument('-D', '--distance', help='The distance measure used to calculate MSV.', type=str, choices=['H', 'C'], required=True, default='H') # H for hellinger and C for chernoff

    args = parser.parse_args()
    print(args)

    N = args.N 
    num_trials = args.num_trials
    dataset = args.dataset
    distance = args.distance


    if dataset == 'BreastCancer':
        X, Y = load_breast_cancer(return_X_y=True)
    elif dataset == 'CovType':
        X, Y = fetch_covtype(return_X_y=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


    curves_for_increasing_SV, curves_for_decreasing_SV = [], []

    exp_dir = oj('RandomForestPruning_results', args.dataset, f'N{str(N)}depth{str(args.max_depth)}D{args.distance}')

    for m in range(num_trials):

        trial_dir = oj(exp_dir, 'trial-{}-of-{}'.format(str(m+1), str(num_trials)))
        os.makedirs(trial_dir, exist_ok=True)
        
        with cwd(trial_dir):

            clf = RandomForestClassifier(max_depth=args.max_depth, random_state=m, n_estimators=N)
            clf.fit(X_train, Y_train)
            print("Initial score:", clf.score(X_test, Y_test))


            alphas = [get_alpha(base_est, X_train, Y_train) for base_est in clf.estimators_]
            true_alpha = get_alpha(clf, X_train, Y_train)


            alphas = np.asarray(alphas)
            true_alpha = np.asarray(true_alpha)

            np.savetxt('model_alphas', alphas)
            np.savetxt('true_alpha', true_alpha)

            n_cores = multiprocessing.cpu_count() - 4 

            with Pool(n_cores) as pool:
                input_arguments = [(i, alphas, true_alpha)  for i in range(N)]
                output = pool.starmap(get_SV_permu_count, input_arguments)

            SVs_H = np.zeros(N)
            SVs_C = np.zeros(N)
            for (i, sv_H, sv_C) in output:
                SVs_H[i] = sv_H
                SVs_C[i] = sv_C

            # results_dir = oj('{}_results'.format('RandomForestPruning'), dataset, 'N{}'.format(N))
            # save_results(results_dir, SVs_H, SVs_C, alphas)
            # np.savetxt(oj(results_dir, 'alphas'), alphas)

            all_base_estimators = deepcopy(clf.estimators_)

            if distance == 'C':
                zipped_SV_estimators = zip(SVs_C, all_base_estimators)
                sorted_pairs = sorted(zipped_SV_estimators)

                tuples = zip(*sorted_pairs)
                SVs_C, sorted_estimators = [list(tuple) for tuple in  tuples]

            else:

                zipped_SV_estimators = zip(SVs_H, all_base_estimators)
                sorted_pairs = sorted(zipped_SV_estimators)

                tuples = zip(*sorted_pairs)
                SVs_H, sorted_estimators = [list(tuple) for tuple in  tuples]

            # ### Analyzing the performance of adding more base estimators according to their SV

            curve_for_increasing_SV, curve_for_decreasing_SV = [], []
            for index in range(1, N+1):
                clf.estimators_ = sorted_estimators[:index]
                curve_for_increasing_SV.append(clf.score(X_test, Y_test))

                clf.estimators_ = sorted_estimators[::-1][:index]
                curve_for_decreasing_SV.append(clf.score(X_test, Y_test))


            curves_for_increasing_SV.append(curve_for_increasing_SV)
            curves_for_decreasing_SV.append(curve_for_decreasing_SV)


    with cwd(exp_dir):

        curves_for_increasing_SV = np.asarray(curves_for_increasing_SV)
        curves_for_decreasing_SV = np.asarray(curves_for_decreasing_SV)

        np.savetxt(f'curves_for_increasing_SV-{args.distance}', curves_for_increasing_SV)
        np.savetxt(f'curves_for_decreasing_SV-{args.distance}', curves_for_decreasing_SV)

        curves_for_increasing_SV_avg = curves_for_increasing_SV.mean(axis=0)
        curves_for_increasing_SV_stderr = sem(curves_for_increasing_SV, axis=0)
        
        curves_for_decreasing_SV_avg = curves_for_decreasing_SV.mean(axis=0)
        curves_for_decreasing_SV_stderr = sem(curves_for_decreasing_SV, axis=0)

        x = np.arange(N)

        fig = plt.figure(figsize=(8, 6))

        plt.errorbar(x, curves_for_increasing_SV_avg, yerr=curves_for_increasing_SV_stderr,  label='Lowest SV first', linewidth=4)
        plt.errorbar(x, curves_for_decreasing_SV_avg, yerr=curves_for_decreasing_SV_stderr,  label='Highest SV first', linewidth=4)

        # plt.plot(increasing_scores, label='Lowest SV first', linewidth=4)
        # plt.plot(decreasing_scores, label='Highest SV first', linewidth=4)
        plt.legend(loc='lower right')
        plt.xlabel("No. base estimators")
        plt.ylabel("Test Accuracy")
        plt.tight_layout()
        # plt.show()
        # exit()
        plt.savefig(f'accu_vs_estimators_{args.dataset}_M{args.num_trials}_D{args.distance}.png', bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()







