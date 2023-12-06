#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from torchensemble import VotingClassifier
from torchensemble.utils.logging import set_logger
from torchensemble.utils import io # for reload the saved model

from copy import deepcopy
from os.path import join as oj
import os
import argparse

from collections import OrderedDict
from multiprocessing import Pool
import multiprocessing

class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2() 
        self.c2_2 = C2() 
        self.c3 = C3() 
        self.f4 = F4() 
        self.f5 = F5() 

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from utils import get_mle

def test_models_vote(models, test_loader, criterion=nn.CrossEntropyLoss()):
    device = torch.device('cuda')

    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            predicted_proba = []
            for model in models:
                model.eval()
                model = model.to(device)
                
                output = model(data)
                predicted_proba.append(F.softmax(output, dim=1))
            predicted_proba = torch.stack(predicted_proba)
            predicted_proba = torch.mean(predicted_proba, dim=0)

            test_loss += criterion(predicted_proba, target).item()  # sum up batch loss

            pred = predicted_proba.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return 100. * correct / len(test_loader.dataset)


from dirichlet import mle
def get_test_loader_alpha(test_loader, num_classes=10):
    temp = []
    for data, labels in test_loader:
        a = F.one_hot(labels, num_classes=num_classes).float()

        # softmax to get the true alpha
        a = torch.softmax(10*a, dim=1)
        
        temp.append(a.detach().cpu())

    temp = torch.vstack(temp).numpy()

    return mle(temp, method='fixpoint')


def get_pruning_curves(sorted_estimators, test_loader):
    
    N = len(sorted_estimators)

    increasing_scores, decreasing_scores = [], []
    for index in range(1, N+1):

        test_acc = test_models_vote(sorted_estimators[:index], test_loader)    
        increasing_scores.append(test_acc)

        test_acc = test_models_vote(sorted_estimators[::-1][:index], test_loader)
        decreasing_scores.append(test_acc)
    return increasing_scores, decreasing_scores


def estimate_SVs(alphas, true_alpha, N, n_cores = multiprocessing.cpu_count() - 4):

    with Pool(n_cores) as pool:

        input_arguments = [(i, alphas, true_alpha)  for i in range(N)]
        output = pool.starmap(get_SV_permu_count, input_arguments)

    SVs_H = np.zeros(N)
    SVs_C = np.zeros(N)
    for (i, sv_H, sv_C) in output:
        SVs_H[i] = sv_H
        SVs_C[i] = sv_C

    return SVs_H, SVs_C

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

from get_MSVs import get_SV_permu_count

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='Process which dataset to construct a deep ensemble for pruning.')
    parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, default='MNIST')
    parser.add_argument('-N', '--N', help='The number of base estimators', type=int, default=5)
    parser.add_argument('-E', '--E', help='The number of epochs each base estimator runs', type=int, default=10)
    parser.add_argument('-M', '--num_trials', help='The number of random trials to run the estimator curves.', type=int, default=10)
    parser.add_argument('-model', '--model', help='The selected model architecture for training. Only used for CIFAR-100', type=str, default='ResNet')

    parser.add_argument('-D', '--distance', help='The distance measure used to calculate MSV.', type=str, choices=['H', 'C'], default='H') # H for hellinger and C for chernoff


    args = parser.parse_args()
    print(args)
    
    N = args.N
    E = args.E
    num_trials = args.num_trials
    distance = args.distance

    if args.dataset == 'MNIST':
        from mnist_utils import get_loaders
        # from run_mnist import MLP_Net
        base_estimator_class = LeNet5
        num_classes = 10

    elif args.dataset == 'CIFAR-10':
        from cifar10_utils import get_loaders
        base_estimator_class = LeNet
        num_classes = 10


    elif args.dataset == 'CIFAR-100':
        base_estimator_class = models.resnet18(pretrained=True)
        base_estimator_class.fc = nn.Linear(512, 100)
        base_estimator_class.__name__ = 'ResNet18'
        num_classes = 100

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
    
    # Define the ensemble
    model = VotingClassifier(
        estimator=base_estimator_class,
        n_estimators=N,
        cuda=True,
    )


    save_dir = oj('DeepEnsemblePruning_results', args.dataset, f'N{str(N)}E{str(E)}D{distance}' )
    os.makedirs(save_dir, exist_ok=True)
    
    os.chdir(save_dir)

    # Set the criterion
    criterion = nn.CrossEntropyLoss()
    model.set_criterion(criterion)

    logger = set_logger('{}_{}'.format(args.dataset, base_estimator_class.__name__))
    # Set the optimizer
    if args.dataset == 'MNIST':
        model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

    elif args.dataset == 'CIFAR-10':
        model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

    elif args.dataset == 'CIFAR-100':
        model.set_optimizer('SGD', lr=1e-1, weight_decay=5e-4, momentum=0.9)

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

    ## Computing $\alpha$ and analyzing SV

    try:
        alphas = np.loadtxt('model_alphas')
        true_alpha = np.loadtxt('true_alpha')
        print('Loading alphas successful.')
    except Exception as e:
        print(str(e))

        alphas = []
        for base_est in model.estimators_:
            alphas.append(get_mle(base_est, test_loader, num_classes))

        # true_alpha = get_mle(model, test_loader, num_classes)
        if args.dataset == 'CIFAR-100':
            true_alpha = get_test_loader_alpha(test_loader, num_classes=100)
        else:
            true_alpha = get_test_loader_alpha(test_loader, num_classes=10)

        alphas = np.asarray(alphas)
        true_alpha = np.asarray(true_alpha)

        np.savetxt('model_alphas', alphas)
        np.savetxt('true_alpha', true_alpha)


    curves_for_increasing_SV, curves_for_decreasing_SV = [], []
    for m in range(num_trials):

        # ### Estimating the SV values

        SVs_H, SVs_C = estimate_SVs(alphas, true_alpha, N)

        # ### Ordering the base estimators according to SV

        all_base_estimators = deepcopy(model.estimators_)

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
        curve_for_increasing_SV, curve_for_decreasing_SV = get_pruning_curves(sorted_estimators, test_loader)

        curves_for_increasing_SV.append(curve_for_increasing_SV)
        curves_for_decreasing_SV.append(curve_for_decreasing_SV)


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


