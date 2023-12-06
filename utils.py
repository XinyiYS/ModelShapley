from functools import reduce
import operator

from math import lgamma, gamma, sqrt
import numpy as np


# Distance based
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def Bhatt_Coeff(A, A_):

    C1 = prod([ gamma( (a+a_)/2.0) for a,a_ in zip(A,A_)  ])

    C2 = gamma(0.5* sum(A)+0.5*sum(A_) )

    C3 = sqrt(gamma(sum(A)) * gamma(sum(A_)))

    C4 = sqrt(prod([gamma(a)*gamma(a_) for a,a_ in zip(A,A_)]))

    # print("C1 C2 C3 C4:", C1, C2, C3, C4, "BC is:", (C1 / C4) * (C3/C2))

    return (C1 /C2) * (C3/C4)

def Chernoff_dist(A, A_, l=0.5):

    C1 = lgamma(l * sum(A) + (1-l)* sum(A_))

    C2 = l * sum([ lgamma(alpha)     for alpha in A ]) + (1-l) * sum([ lgamma(alpha) for alpha in A_ ])

    C3 = - ( sum( [ lgamma(l*alpha + (1-l)*alpha_)  for alpha, alpha_ in zip(A,A_)] ))

    C4 = - l * lgamma(sum(A)) - (1-l)*  lgamma(sum(A_))

    # print('Chernoff between', A, A_, C1 + C2 + C3 + C4)

    return C1 + C2 + C3 + C4

def Hellinger_dist(A, A_):
    BC = np.exp(-Chernoff_dist(A, A_))
    return sqrt(2 * max(1 - BC, 1e-8) )
    # return sqrt(2* (1-Bhatt_Coeff(A, A_)))


# SV related
from math import factorial as fac
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))



import numpy as np
from dirichlet import mle # from the installed package at https://github.com/ericsuh/dirichlet.git

import torch
import torch.nn.functional as F

def get_mle_augmented(model, test_loader, augment_precision, device = torch.device('cuda')):
    '''
    Obtain the maximum likelihood estimate for the alpha parameters for the Dirichlet abstraction of a model, with artificially augmented precision.

    '''

    D = []
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            output = torch.softmax(output, dim=1)

            view = output.view(len(output), -1)
            max_indices = ((view==view.max(dim=1, keepdim=True)[0]).view_as(output))
            output += augment_precision *  torch.mul(output, max_indices)

            output = F.normalize(output, p=1, dim = 1)

            # preventing degeneracy in MLE
            output = torch.clamp(output, min=1e-6)
            output = F.normalize(output, p=1, dim=1)

            D.append(output)

    D = torch.vstack(D).detach().cpu().numpy()
    return mle(D, method='fixpoint')


def get_mle(model, test_loader, num_classes, augment_precision=0, device=torch.device('cuda')):
    '''
    Obtain the maximum likelihood estimate for the alpha parameters for the Dirichlet abstraction of a model.

    # Parameters
        models (Pytorch model): the model for which to obtain the Dirichlet abstraction

        test_loader (Pytorch Dataloader): the task, i.e., the data loader for this task (e.g., validation loader)

        num_classes (integer): the nuumber of classes

        augment_precision (float): how much to artifically increase the predictive certainty of the model (i.e., the precision of the Dirichlet abstraction), only used for the precision experiments.

        device (Pytorch device): the device to use for model inferece.

    # Returns

        Alphas (a list/vector of floats of length num_classes): the vector of alpha values representing parametrizing a Dirichlet distribution/abstraction.
    '''

    if augment_precision > 0:
        return get_mle_augmented(model, test_loader, augment_precision)

    model.eval()
    model = model.to(device)

    D = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            
            output = torch.softmax(output, dim=1)

            # preventing degeneracy in MLE
            output = torch.clamp(output, min=1e-6 )
            output = F.normalize(output, p=1, dim=1)

            D.append(output)            

    D = torch.vstack(D).detach().cpu().numpy()
    if len(D) < 5:
        # NOT ENOUGH query/samples for an MLE  set by default
        return torch.ones(num_classes) / num_classes
    else:
        return mle(D, method='fixpoint')


def get_model_alphas_by_class(models, test_loader, num_classes, device = torch.device('cuda')):
    
    '''
    Obtain the Alpha parameters for a list of models, based on a single task partitioned according to the classes.

    '''

    model_alphas_by_class_across_models = []

    for model in models:

        model_alpha_by_class = []
        exp_logits_by_classes = [ [ ] for _ in range(num_classes)]

        model.eval()
        model = model.to(device)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                exp_logits = torch.softmax(output, dim=1)

                # preventing degeneracy in MLE
                exp_logits = torch.clamp(exp_logits, min=1e-4)
                exp_logits = F.normalize(exp_logits, p=1, dim=1)

                for label in range(num_classes):
                    exp_logits_by_classes[label].append(exp_logits[target==label])

        for exp_logits_by_class in exp_logits_by_classes:
            exp_logits_by_class = torch.vstack(exp_logits_by_class).detach().cpu().numpy()
        
            if len(exp_logits_by_class) < 5:
                # NOT ENOUGH query/samples for an MLE  set by default
                alpha_by_class = np.ones(num_classes) / num_classes
            else:
                try:
                    alpha_by_class = mle(exp_logits_by_class, method='fixpoint')
                except Exception as e:
                    # print("fixpoint faied trying mean precision instead.")
                    try:
                        alpha_by_class = mle(exp_logits_by_class, method='meanprecision')
                    except Exception as e:
                        # print("mean precision also faied. by default use uniform")
                        alpha_by_class = np.ones(num_classes) / num_classes


            model_alpha_by_class.append(alpha_by_class)
        
        model_alphas_by_class_across_models.append(np.asarray(model_alpha_by_class))

    return np.asarray(model_alphas_by_class_across_models)


def get_model_alphas(models, eval_loader, num_classes, by_class=False, precise_type=False):
    '''
    Obtain the Alpha parameters for a list of models, based on a single task.

    # Parameters
        models (a list of Pytorch models): the list of model for which to obtain the Dirichlet abstractions

        eval_loader (Pytorch Dataloader): the task, i.e., the data loader for this task (e.g., validation loader)

        num_classes (integer): the nuumber of classes

        by_class (bool): whether to perform by class partition

        precise_type (bool): whether to artificially increase the predictive certainty of the model (i.e., precision of the corresponding Dirichlet abstraction)

    # Returns

        A list of list of floats: A list of length equal to the length of models, and each element is a list of alphas equal to the number of classes.      

        The length of the list may not be equal to the length of models, if precise_type=True and the length of models is not divisible by 3.        
    '''

    if by_class:
        if precise_type:
            raise NotImplementedError("Simultaneously using by-class partition and artificially increasing precision of the models is not implemented. ")
        
        return get_model_alphas_by_class(models, eval_loader, num_classes)

    if precise_type:
        individual_N = len(models) // 3
        
        alphas = [get_mle(model, eval_loader) for model in models[:individual_N]]

        alphas.extend( [get_mle(model, eval_loader, augment_precision=20) for model in models[individual_N:2*individual_N]] )

        alphas.extend( [get_mle(model, eval_loader, augment_precision=50) for model in models[2*individual_N:]] )

        return alphas
    
    else:
        return [get_mle(model, eval_loader) for model in models]

def get_test_loader_alpha_by_class(test_loader, num_classes, noise=0.05):

    from scipy.special import softmax
    true_alpha_by_class = []
    
    for label in range(num_classes):
        D = []
        for data, target in test_loader:
            one_hot_by_class = F.one_hot(target[target==label], num_classes=num_classes).float()
            one_hot_by_class += torch.rand_like(one_hot_by_class) * noise # use 0.05 for models variation, o/w 0.01
            one_hot_by_class =  F.normalize(one_hot_by_class, p=1, dim=1)


            # preventing degeneracy in MLE
            one_hot_by_class = torch.clamp(one_hot_by_class, min=1e-6)
            one_hot_by_class = F.normalize(one_hot_by_class, p=1, dim=1)

            D.append(one_hot_by_class)

        D = torch.vstack(D).detach().cpu().numpy()

        if len(D) < 5:
            # NOT ENOUGH query/samples for an MLE  set by default
            true_alpha_by_class.append( torch.ones(num_classes) / num_classes)
        else:
            true_alpha_by_class.append(mle(D, method='fixpoint'))

    return true_alpha_by_class

def get_test_loader_alpha(test_loader, num_classes, by_class=False,  noise=0.05):
    '''
    This is for Q* by enumerating the (X,y) pairs from a test loader, and is different from constructing the Dirichlet abstraction for a model.
    We directly use the mle() function from the dirichlet package.

    # Parameters
        test_loader (Pytorch DataLoader): a loader based on from the task (e.g., test dataset)

        num_classes (integer): the nuumber of classes

        by_class (bool): whether to perform by class partition

        noise (float): a small amount of uniform noise to prevent numerical degeneracy during the maximum likelihood estimation (MLE)
    
    # Returns
        true_alphas (a list/vector of floats of length num_classes): the vector of alpha values representing parametrizing a Dirichlet distribution/abstraction 
    '''

    if by_class: 
        return get_test_loader_alpha_by_class(test_loader=test_loader, num_classes=num_classes, noise=0.05)

    temp = []
    for data, labels in test_loader:

        a = F.one_hot(labels, num_classes=num_classes).float()
        a += torch.rand_like(a) * noise # use 0.05 for models variation, o/w 0.01
        a = F.normalize(a, p=1, dim = 1)

        # preventing degeneracy in MLE
        a = torch.clamp(a, min=1e-6)
        a = F.normalize(a, p=1, dim=1)

        temp.append(a.detach().cpu())

    temp = torch.vstack(temp).numpy()
    true_alpha = mle(temp, method='fixpoint')
    return true_alpha


from sklearn.metrics import f1_score
def get_f1_score(model, device, test_loader):
    model.eval()
    model = model.to(device)
    
    f1_score_avg = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            target, pred = target.detach().cpu().numpy(), pred.detach().cpu().numpy()
            f1_score_avg += len(target) * f1_score(target, pred, average='macro')

    return round(f1_score_avg / len(test_loader.dataset), 4)


from typing import Iterable
'''
Precision-weighted fusion
'''
def precision_fusion(alphas: Iterable[Iterable]) -> list:
    '''
    Takes in a list of Dirichlet (abstractions), each parametrized by a list of alpha parameters of the same dimension D.

    Returns a single Dirichlet abstraction, parametrized by a list of alpha parameters of dimension D.
    '''
    fused_alpha = np.sum(np.asarray(alphas), axis=0)
    # fused_alpha = np.mean(np.asarray(alphas), axis=0)
    return fused_alpha

# P_fused_alpha = precision_fusion([[1, 1, 1], [1, 1, 20]])
# P_fused_dir = dirichlet(P_fused_alpha)
# P_fused_dir.pdf([0.1,0.1,0.8])


import os
from contextlib import contextmanager

@contextmanager
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

import numpy as np
from os.path import join as oj
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
if find_executable('latex'): 
    print('latex installed')
    plt.rcParams['text.usetex'] = True

import seaborn as sns

def save_results(results_dir, SVs_H, SVs_C, model_alphas, by_class=False, class_weights=[], mistake_accuracies=[], labels = [], f1_scores=[]):

    '''
    Saving the SV results, pearson correlation coefficients and plots.
    
    '''

    os.makedirs(results_dir, exist_ok=True)

    N = len(SVs_H)
    initial_distances_H = np.zeros((N,N))
    initial_distances_C = np.zeros((N,N))
    SV_diffs_H =  np.zeros((N,N))
    SV_diffs_C =  np.zeros((N,N))

    if by_class:
        for i in range(N):
            for j in range(N):
                for label, class_weight in enumerate(class_weights):                    
                    initial_distances_H[i,j] = Hellinger_dist(model_alphas[i][label], model_alphas[j][label]) * class_weight
                    initial_distances_C[i,j] = Chernoff_dist(model_alphas[i][label], model_alphas[j][label]) * class_weight
                    
                SV_diffs_H[i,j] = abs(SVs_H[i]- SVs_H[j])
                SV_diffs_C[i,j] = abs(SVs_C[i]- SVs_C[j])

    else:

        for i in range(N):
            for j in range(N):
                initial_distances_H[i,j] = Hellinger_dist(model_alphas[i], model_alphas[j])
                initial_distances_C[i,j] = Chernoff_dist(model_alphas[i], model_alphas[j])
                SV_diffs_H[i,j] = abs(SVs_H[i]- SVs_H[j])
                SV_diffs_C[i,j] = abs(SVs_C[i]- SVs_C[j])


    os.makedirs(results_dir, exist_ok=True)

    np.savetxt(oj(results_dir, 'initial_distances_H'), initial_distances_H)
    np.savetxt(oj(results_dir, 'initial_distances_C'), initial_distances_C)
    np.savetxt(oj(results_dir, 'SVs_H'), SVs_H)
    np.savetxt(oj(results_dir, 'SVs_C'), SVs_C)
    np.savetxt(oj(results_dir, 'SV_diffs_H'), SV_diffs_H)
    np.savetxt(oj(results_dir, 'SV_diffs_C'), SV_diffs_C)
    

    coeff_H = pearsonr(initial_distances_H.flatten(), SV_diffs_H.flatten())
    coeff_C = pearsonr(initial_distances_C.flatten(), SV_diffs_C.flatten())
    with open(oj(results_dir, 'pearson coeff'), 'w') as file:
        
        file.write('Hellinger:   ')
        file.write(str(coeff_H))
        file.write( '\n')
        file.write('Chernoff:   ')
        file.write(str(coeff_C))

    if N <= 15:

        ind = np.arange(N)  # the x locations for the groups
        barWidth = 3.0 / N

        fig = plt.figure(figsize=(6, 4)) # Create matplotlib figure
        ax1 = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.

        # rects1 = ax1.bar(ind, SVs_C, color='C0', edgecolor='black', width=barWidth, label='MSV')
        # rects2 = ax2.bar(ind+barWidth, mistake_accuracies, color='C1', edgecolor='black', width=barWidth, label='Accuracy')
        rects1 = ax1.bar(ind, SVs_C, color='C0', width=barWidth, label='MSV')
        rects2 = ax2.bar(ind+barWidth, mistake_accuracies, color='C1', width=barWidth, label='Accuracy')

        ax2.legend( (rects1[0], rects2[0]), ('MSV', 'Accuracy') , loc='lower right')

        ax1.set_ylabel('MSV')
        ax2.set_ylabel('Accuracy')
        ax1.grid(False)
        ax2.grid(False)

        # labels = ['CNN', 'MLP', 'LR']
        if labels:
            label_pos = [ (0.5 + i) * N // len(labels) for i in range(len(labels))]
            plt.xticks(ticks=label_pos, labels=labels)
        else:
            plt.xticks(ticks=np.arange(N)+0.5*barWidth, labels=np.arange(N))

    else:
        fig = plt.figure(figsize=(6, 4)) # Create matplotlib figure
        ax1 = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.

        line1 = ax1.plot(SVs_C, color='C0', label='MSV')
        line2 = ax2.plot(mistake_accuracies, color='C1', label='Accuracy')

        ax2.legend( (rects1[0], rects2[0]), ('MSV', 'Accuracy') , loc='lower right')
        ax1.set_ylabel('MSV')
        ax2.set_ylabel('Accuracy')
        if labels:
            label_pos = [ (0.5 + i) * N // len(labels) for i in range(len(labels))]
            plt.xticks(ticks=label_pos, labels=labels)
        else:
            plt.xticks(ticks=np.arange(N), labels=np.arange(N))

    plt.tight_layout()
    # plt.show()
    plt.savefig(oj(results_dir, 'SVs_C.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


    if N <= 15:
        ind = np.arange(N)  # the x locations for the groups
        barWidth = 3.0 / N

        fig = plt.figure(figsize=(6, 4)) # Create matplotlib figure
        ax1 = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.

        # rects1 = ax1.bar(ind, SVs_H, color='C0', edgecolor='black', width=barWidth, label='MSV')
        # rects2 = ax2.bar(ind+barWidth, mistake_accuracies, color='C1', edgecolor='black', width=barWidth, label='Accuracy')

        rects1 = ax1.bar(ind, SVs_H, color='C0', width=barWidth, label='MSV')
        rects2 = ax2.bar(ind+barWidth, mistake_accuracies, color='C1', width=barWidth, label='Accuracy')


        ax2.legend( (rects1[0], rects2[0]), ('MSV', 'Accuracy') , loc='lower right')

        ax1.set_ylabel('MSV')
        ax2.set_ylabel('Accuracy')
        ax1.grid(False)
        ax2.grid(False)

        if labels:
            label_pos = [ (0.5 + i) * N // len(labels) for i in range(len(labels))]
            plt.xticks(ticks=label_pos, labels=labels)
        else:
            plt.xticks(ticks=np.arange(N)+0.5*barWidth, labels=np.arange(N))
        
    else:
        fig = plt.figure(figsize=(6, 4)) # Create matplotlib figure
        ax1 = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.

        line1 = ax1.plot(SVs_H, color='C0', label='MSV')
        line2 = ax2.plot(mistake_accuracies, color='C1', label='Accuracy')

        ax2.legend( (line1[0], line2[0]), ('MSV', 'Accuracy'), loc='lower right')

        ax1.set_ylabel('MSV')
        ax2.set_ylabel('Accuracy')
        if labels:
            label_pos = [ (0.5 + i) * N // len(labels) for i in range(len(labels))]
            plt.xticks(ticks=label_pos, labels=labels)
        else:
            plt.xticks(ticks=np.arange(N), labels=np.arange(N))

    plt.tight_layout()
    # plt.show()
    plt.savefig(oj(results_dir, 'SVs_H.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


    # additional f1 score plots

    if len(f1_scores) > 0:

        ind = np.arange(N)  # the x locations for the groups
        barWidth = 3.0 / N

        fig = plt.figure(figsize=(6, 4)) # Create matplotlib figure
        ax1 = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.

        rects1 = ax1.bar(ind, SVs_C, color='C0', width=barWidth, label='MSV')
        rects2 = ax2.bar(ind+barWidth, f1_scores, color='C1', width=barWidth, label='F1 score')

        ax2.legend( (rects1[0], rects2[0]), ('MSV', 'F1 score') , loc='lower right')

        ax1.set_ylabel('MSV')
        ax2.set_ylabel('F1 score')
        ax1.grid(False)
        ax2.grid(False)

        # labels = ['CNN', 'MLP', 'LR']
        if labels:
            label_pos = [ (0.5 + i) * N // len(labels) for i in range(len(labels))]
            plt.xticks(ticks=label_pos, labels=labels)
        else:
            plt.xticks(ticks=np.arange(N)+0.5*barWidth, labels=np.arange(N))

        plt.tight_layout()
        # plt.show()
        plt.savefig(oj(results_dir, 'SVs_C_F1.png'), bbox_inches='tight')
        plt.clf()
        plt.close()



        ind = np.arange(N)  # the x locations for the groups
        barWidth = 3.0 / N

        fig = plt.figure(figsize=(6, 4)) # Create matplotlib figure
        ax1 = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.

        rects1 = ax1.bar(ind, SVs_H, color='C0', width=barWidth, label='MSV')
        rects2 = ax2.bar(ind+barWidth, f1_scores, color='C1', width=barWidth, label='F1 score')

        ax2.legend( (rects1[0], rects2[0]), ('MSV', 'F1 score') , loc='lower right')

        ax1.set_ylabel('MSV')
        ax2.set_ylabel('F1 score')
        ax1.grid(False)
        ax2.grid(False)

        # labels = ['CNN', 'MLP', 'LR']
        if labels:
            label_pos = [ (0.5 + i) * N // len(labels) for i in range(len(labels))]
            plt.xticks(ticks=label_pos, labels=labels)
        else:
            plt.xticks(ticks=np.arange(N)+0.5*barWidth, labels=np.arange(N))

        plt.tight_layout()
        # plt.show()
        plt.savefig(oj(results_dir, 'SVs_H_F1.png'), bbox_inches='tight')
        plt.clf()
        plt.close()

    sns.set(font_scale=2)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6, 2.25)) # figsize=(16, 6)
    sns.heatmap(initial_distances_H, ax=ax1)
    sns.heatmap(SV_diffs_H, ax=ax2)
    # plt.suptitle("Hellinger valuation: $d_{H}(i,i')$ vs. $|\phi_i - \phi_{i'}|$", fontsize=32)

    plt.tight_layout()
    plt.savefig(oj(results_dir, 'H.png'), bbox_inches='tight')
    # plt.show()
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6, 2.25))
    sns.heatmap(initial_distances_C, ax=ax1)
    sns.heatmap(SV_diffs_C, ax=ax2)
    # plt.suptitle("Chernoff valuation: $d_{C}(i,i')$ vs. $|\phi_i - \phi_{i'}|$", fontsize=32)
    # plt.show()
    plt.tight_layout()
    plt.savefig(oj(results_dir, 'C.png'), bbox_inches='tight')
    plt.clf()
    plt.close()
    return
