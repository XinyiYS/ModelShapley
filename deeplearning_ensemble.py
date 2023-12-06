#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

from torchensemble import VotingClassifier, SnapshotEnsembleClassifier
from torchensemble.utils.logging import set_logger
from torchensemble.utils import io # for reload the saved model

import torchvision.models as models

from copy import deepcopy
from os.path import join as oj
import os
import argparse

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

from scipy.stats import sem

from utils import get_mle, cwd

from deeplearning_ensemble_SV_multiple import get_pruning_curves, estimate_SVs, get_test_loader_alpha, LeNet, LeNet5


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process which dataset to construct a deep ensemble for pruning.')
    parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, default='MNIST')
    parser.add_argument('-N', '--N', help='The number of base estimators', type=int, default=5)
    parser.add_argument('-E', '--E', help='The number of epochs each base estimator runs', type=int, default=10)
    parser.add_argument('-M', '--num_trials', help='The number of random trials', type=int, default=10)

    parser.add_argument('-model', '--model', help='The selected model architecture for training. Only used for CIFAR-100', type=str, default='ResNet')

    args = parser.parse_args()
    print(args)
    
    N = args.N
    E = args.E
    num_trials = args.num_trials
    
    if args.dataset == 'MNIST':
        from mnist_utils import get_loaders
        base_estimator_class = LeNet5

    elif args.dataset == 'CIFAR-10':
        from cifar10_utils import get_loaders
        base_estimator_class = LeNet

    elif args.dataset == 'CIFAR-100':
        # base_estimator_class = models.resnet18(pretrained=True)
        # base_estimator_class.fc = nn.Linear(512, 100)
        # base_estimator_class.__name__ = 'ResNet18'
        from cifar100_utils import get_loaders

        if args.model == 'Efficient':
            base_estimator_class = models.efficientnet_b7(pretrained=True)        
            num_ftrs = base_estimator_class.classifier[1].in_features
            base_estimator_class.classifier[1] = nn.Linear(num_ftrs, 100)
            base_estimator_class.__name__ = 'EfficientNetB7'

        elif args.model == 'Dense':
            base_estimator_class = models.densenet121(pretrained=True)        
            num_ftrs = base_estimator_class.classifier.in_features
            base_estimator_class.classifier = nn.Linear(num_ftrs, 100)
            base_estimator_class.__name__ = 'DenseNet121'
        else:
            base_estimator_class = models.resnet18(pretrained=True)
            base_estimator_class.fc = nn.Linear(512, 100)
            base_estimator_class.__name__ = 'ResNet18'

    train_loader, test_loader = get_loaders()
    
    criterion = nn.CrossEntropyLoss()
    curves_for_increasing_SV, curves_for_decreasing_SV = [], []

    exp_dir = oj('DeepEnsemblePruning_results', args.dataset, 'N{}E{}'.format(str(N), str(E)))
    for m in range(num_trials):
        # Define the ensemble
        model = VotingClassifier(
            estimator=deepcopy(base_estimator_class),
            n_estimators=N,
            cuda=True,
        )

        trial_dir = oj(exp_dir, 'trial-{}-of-{}'.format(str(m+1), str(num_trials)))
        os.makedirs(trial_dir, exist_ok=True)
        
        # os.chdir(trial_dir)
        with cwd(trial_dir):

            # Set the criterion
            model.set_criterion(criterion)

            # Set the optimizer
            if args.dataset == 'MNIST':
                model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
                logger = set_logger('MNIST_LeNet5')

            elif args.dataset == 'CIFAR-10':
                model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
                logger = set_logger('CIFAR-10_LeNet5')

            elif args.dataset == 'CIFAR-100':
                model.set_optimizer('SGD', lr=1e-1, weight_decay=5e-4, momentum=0.9)

                if args.model  == 'Efficient':
                    logger = set_logger('CIFAR-100_EfficientNetB7')
        
                else:
                    logger = set_logger('CIFAR-100_DenseNet121')

                model.set_scheduler(
                    "MultiStepLR",                    # type of learning rate scheduler
                    milestones =  [150, 225],
                    gamma = 0.1
                )

            # Train and Evaluate
            try:
                io.load(model)  # reload
                print("Loading trained ensemble successful.")

            except Exception as e:
                print(str(e))
                # Train and Evaluate
                model.fit(
                    train_loader,
                    epochs=E,
                    test_loader=test_loader,
                )

            # ## Computing $\alpha$ and analyzing SV
            try:
                alphas = np.loadtxt('model_alphas')
                true_alpha = np.loadtxt('true_alpha')
                print('Loading alphas successful.')
            except Exception as e:
                print(str(e))

                alphas = []
                for base_est in model.estimators_:
                    alphas.append(get_mle(base_est, test_loader))

                # true_alpha = get_mle(model, test_loader)
                if args.dataset == 'CIFAR-100':
                    true_alpha = get_test_loader_alpha(test_loader, num_classes=100)
                else:
                    true_alpha = get_test_loader_alpha(test_loader, num_classes=10)

                alphas = np.asarray(alphas)
                true_alpha = np.asarray(true_alpha)

                np.savetxt('model_alphas', alphas)
                np.savetxt('true_alpha', true_alpha)

            SVs_H, SVs_C = estimate_SVs(alphas, true_alpha, N)

            # ### Ordering the base estimators according to SV
            all_base_estimators = deepcopy(model.estimators_)
            zipped_SV_estimators = zip(SVs_C, all_base_estimators)
            sorted_pairs = sorted(zipped_SV_estimators)

            tuples = zip(*sorted_pairs)
            SVs_C, sorted_estimators = [list(tuple) for tuple in  tuples]

            # ### Analyzing the performance of adding more base estimators according to their SV
            curve_for_increasing_SV, curve_for_decreasing_SV = get_pruning_curves(sorted_estimators, test_loader)

            curves_for_increasing_SV.append(curve_for_increasing_SV)
            curves_for_decreasing_SV.append(curve_for_decreasing_SV)


            del model

    with cwd(exp_dir):

        curves_for_increasing_SV = np.asarray(curves_for_increasing_SV)
        curves_for_decreasing_SV = np.asarray(curves_for_decreasing_SV)

        np.savetxt('curves_for_increasing_SV', curves_for_increasing_SV)
        np.savetxt('curves_for_decreasing_SV', curves_for_decreasing_SV)

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
        plt.savefig('accu_vs_estimators_{}_M{}.png'.format(args.dataset, args.num_trials), bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()



