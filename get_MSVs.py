import os
from os.path import join as oj
import argparse
import random
from math import factorial as fac

import numpy as np
import torch

import seaborn as sns; sns.set_theme()


from utils import powerset, Hellinger_dist, Chernoff_dist, precision_fusion, save_results

def get_SV_permu_count(i, model_alphas, true_alpha, sampling_cost=3745):

    N = len(model_alphas)
    permutation = list(range(N))

    SV_H, SV_C = 0, 0
    for _ in range(sampling_cost):
        while True:
            # explicitly avoiding empty permutation
            random.shuffle(permutation)
            
            if permutation[0] != i: 
                break

        pos = permutation.index(i)

        ensemble_before = precision_fusion([model_alphas[j] for j in permutation[:pos] ] )

        ensemble_after = precision_fusion([model_alphas[j] for j in permutation[:pos + 1] ] )

        marginal = -Hellinger_dist(ensemble_after, true_alpha) - (-Hellinger_dist(ensemble_before, true_alpha))
        SV_H +=  marginal /  sampling_cost

        marginal = -Chernoff_dist(ensemble_after, true_alpha) - (-Chernoff_dist(ensemble_before, true_alpha))
        SV_C +=  marginal /  sampling_cost

    return i, SV_H, SV_C


def get_SV_permu_count_by_class(i, model_alphas_by_class, true_alpha_by_class, class_weights, sampling_cost=3745):

    N = len(model_alphas_by_class)
    permutation = list(range(N))

    SV_H, SV_C = 0, 0
    for _ in range(sampling_cost):
        while True:
            # explicitly avoiding empty permutation
            random.shuffle(permutation)
            
            if permutation[0] != i: 
                break

        pos = permutation.index(i)


        for label, class_weight in enumerate(class_weights):

            ensemble_before = precision_fusion([model_alphas_by_class[j][label] for j in permutation[:pos] ] )

            ensemble_after = precision_fusion([model_alphas_by_class[j][label] for j in permutation[:pos + 1] ] )

            marginal = -Hellinger_dist(ensemble_after, true_alpha_by_class[label]) - (-Hellinger_dist(ensemble_before, true_alpha_by_class[label]))
            SV_H +=  marginal /  sampling_cost * class_weight

            marginal = -Chernoff_dist(ensemble_after, true_alpha_by_class[label]) - (-Chernoff_dist(ensemble_before, true_alpha_by_class[label]))
            SV_C +=  marginal /  sampling_cost * class_weight

    return i, SV_H, SV_C


from torch.utils.data import Dataset

class MistakeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_mistake_loader(test_loader, models, device=torch.device('cuda'), repeat=True):
    '''
    Collect all the mistakes - data which were misclassified by any of the models into a loader.

    repeat means identical mistakes are allowed to appear: if a datum is misclassified by multiple models

    '''
    mistakes = []
    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)


            if repeat:

                for model in models:

                    model.eval()
                    model = model.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                    mistake_indices = pred.ne(target.view_as(pred)).long()
                
                    mistakes.extend( [(mistake_x, mistake_y) for mistake_x, mistake_y in zip(data[mistake_indices], target[mistake_indices])]                            )

            else:

                mistake_indices = torch.zeros((len(data), 1), device=device)

                for model in models:
                    model.eval()
                    model = model.to(device)
                    output = model(data)

                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                    mistake_indices += pred.ne(target.view_as(pred))
                mistake_indices = (mistake_indices > 0).view(len(data))

                mistakes.extend([(mistake_x, mistake_y) for mistake_x, mistake_y in zip(data[mistake_indices], target[mistake_indices])]                            )

    mistake_set = MistakeDataset(mistakes)
    print("The total number of mistakes: {}, allowing repeated mistakes: {}.".format(len(mistake_set), repeat))
    return torch.utils.data.DataLoader(mistake_set, shuffle=True, batch_size=256)


from utils_ML import test
def get_accuracies(models, eval_loader):
    return [test(model, torch.device('cuda'), eval_loader) for model in models]

from multiprocessing import Pool
import multiprocessing

from utils import get_model_alphas, get_test_loader_alpha, get_f1_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process which dataset to analyze the precision ensemble')
    parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, default='MNIST')
    parser.add_argument('-iN', '--individual_N', help='The integer representing the individual N', type=int, default=3)
    parser.add_argument('-l', '--l', help='The weight coefficient lambda towards optimal in Chernoff distance', type=float, default=0.5)
    parser.add_argument('--by_class', dest='by_class',help='Compute distance by classes for input distribution. Only supported for MNIST and CIFAR-10.', action='store_true')
    parser.add_argument('--not_by_class', dest='by_class', action='store_false')

    parser.add_argument('-t', '--type', help='The type of experiments.', type=str, default='models', choices=['datasets', 'models', 'precise'])

    parser.add_argument('-m', '--eval_mode', help='The evaluation mode, determines the type of loader.', type=str,  default='test', choices=['test', 'mistake'])
    args = parser.parse_args()
    print(args)


    if args.dataset == 'MNIST':
        from mnist_utils import get_loaders, get_models, MODEL_LABELS, DATASIZE_LABELS

    elif args.dataset == 'CIFAR-10':
        from cifar10_utils import get_loaders, get_models, MODEL_LABELS, DATASIZE_LABELS

    elif args.dataset == 'MedNIST':
        from MedNIST_utils import get_loaders, get_models, MODEL_LABELS, DATASIZE_LABELS

    elif args.dataset == 'KDD99':
        from KDD99_utils import get_loaders, get_models, MODEL_LABELS, DATASIZE_LABELS
    
    elif args.dataset == 'DrugReviews':
        from DrugReviews_utils import get_loaders, get_models, MODEL_LABELS, DATASIZE_LABELS

    individual_N = args.individual_N

    # NOTE: need to update the correct directory of the saved_models in the respective `get_models` function
    models = get_models(individual_N, args.type)

    train_loader, test_loader = get_loaders()
    mistake_loader = create_mistake_loader(test_loader, models, repeat=False)

    mistake_accuracies = get_accuracies(models=models, eval_loader=mistake_loader)
    print("Test accuracies on mistakes:", mistake_accuracies)

    mistake_f1_scores = [ get_f1_score(model, torch.device('cuda'), test_loader) for model in models]
    print("F1 scores on mistakes:", mistake_f1_scores)

    test_accuracies = get_accuracies(models=models, eval_loader=test_loader)
    print("True Test accuracies :", test_accuracies)

    test_f1_scores = [ get_f1_score(model, torch.device('cuda'), mistake_loader) for model in models]
    print("True test f1 scores :", test_f1_scores)

    if args.eval_mode == 'mistake':
        '''
        The task/query set consists of the data in the validation that any of the models has made a mistake on (i.e., predicted incorrectly about).

        This is to highlight the difference in the predictions among the models, by excluding the data (in the validation) where all models predicted correctly.
        '''
        eval_loader = mistake_loader
        eval_accuracies = mistake_accuracies
        f1_scores = mistake_f1_scores

    elif args.eval_mode == 'test':
        eval_loader = test_loader
        eval_accuracies = test_accuracies
        f1_scores = test_f1_scores


    NUM_CLASS = {'MNIST':10, 'CIFAR-10':10, 'MedNIST':6, 'KDD99': 23, 'DrugReviews': 14}
    num_classes = NUM_CLASS[args.dataset]

    print("For {}, the number of classes is {}.".format(args.dataset, num_classes))

    if args.by_class:
        true_alpha_by_class = get_test_loader_alpha(test_loader=eval_loader, num_classes=num_classes, by_class=True, noise=0.01)

        model_alphas_by_class = get_model_alphas(models, eval_loader, num_classes=num_classes, by_class=True, precise_type=args.type=='precise')

        # model_alphas_by_class =  np.vstack(different_model_alphas_by_class)

        print("Number of model alphas:", len(model_alphas_by_class), "Shape of alphas:", model_alphas_by_class.shape)
        
        N = len(model_alphas_by_class)

        mistake_count_by_class = [0 for _ in range(num_classes)]

        for data, target in eval_loader:
            for label in range(num_classes):
                mistake_count_by_class[label] += (target == label).sum().item()
    
        class_weights = np.asarray(mistake_count_by_class) / sum(mistake_count_by_class)

        print("Mistake counts by class :{} and the normalized class weights: {}.".format(mistake_count_by_class, class_weights))
        class_weights = np.ones(num_classes) / num_classes

    else:
        true_alpha = get_test_loader_alpha(test_loader=eval_loader, num_classes=num_classes, by_class=False, noise=0.01)

        model_alphas = get_model_alphas(models, eval_loader, num_classes=num_classes, precise_type=args.type=='precise')
        model_alphas = np.asarray(model_alphas)

        # true_alpha = get_test_loader_alpha()
        # different_model_alphas = get_model_alphas(individual_N)
        # model_alphas =  np.vstack(different_model_alphas)

        print("Number of model alphas:", len(model_alphas), "Shape of alphas:", model_alphas.shape)

        N = len(model_alphas)

    if N < 20:
        # exact SV
        SVs_H = np.zeros(N)
        SVs_C = np.zeros(N)
        for i in range(N):
            minus_i= list(range(N))
            minus_i.remove(i)
            coalitions=list(powerset(minus_i))
            for c in coalitions:

                C = len(c)        
                factor = (fac(C) * fac(N - C-1) /fac(N))
                if not c:
                    continue

                else:

                    if not args.by_class:

                        ensemble_before = precision_fusion([model_alphas[j] for j in c ]  )
                        ensemble_after = precision_fusion([model_alphas[j] for j in list(c)+[i]  ]  )

                        marginal = -Hellinger_dist(ensemble_after, true_alpha) - (-Hellinger_dist(ensemble_before, true_alpha))
                        SVs_H[i] +=  factor * marginal

                        marginal = -Chernoff_dist(ensemble_after, true_alpha, args.l) - (-Chernoff_dist(ensemble_before, true_alpha, args.l))
                        SVs_C[i] +=  factor * marginal
                    else:
                        for label, class_weight in enumerate(class_weights):

                            ensemble_before = precision_fusion([model_alphas_by_class[j][label] for j in c ]  )
                            ensemble_after = precision_fusion([model_alphas_by_class[j][label] for j in list(c)+[i]  ]  )

                            marginal = -Hellinger_dist(ensemble_after, true_alpha_by_class[label]) - (-Hellinger_dist(ensemble_before, true_alpha_by_class[label]))
                            SVs_H[i] +=  factor * marginal * class_weight

                            marginal = -Chernoff_dist(ensemble_after, true_alpha_by_class[label], args.l) - (-Chernoff_dist(ensemble_before, true_alpha_by_class[label], args.l))
                            SVs_C[i] +=  factor * marginal * class_weight

    else: # N >= 20
        # multiple-core parallelization for sampling-based approximation
        n_cores = multiprocessing.cpu_count() - 4 

        with Pool(n_cores) as pool:
            if args.by_class:
                input_arguments = [(i, model_alphas_by_class, true_alpha_by_class, class_weights)  for i in range(N)]
                output = pool.starmap(get_SV_permu_count_by_class, input_arguments)

            else:
                input_arguments = [(i, model_alphas, true_alpha)  for i in range(N)]
                output = pool.starmap(get_SV_permu_count, input_arguments)

        SVs_H = np.zeros(N)
        SVs_C = np.zeros(N)
        for (i, sv_H, sv_C) in output:
            SVs_H[i] = sv_H
            SVs_C[i] = sv_C

    # check the initial distance to expert
    if args.by_class:
        distances_to_expert_H = np.zeros(N)
        distances_to_expert_C = np.zeros(N)

        # print("True alpha label for class 0:", true_alpha_by_class[0])
        for i, single_model_alphas_by_class in enumerate(model_alphas_by_class):
            # print("model index: {}, alphas by class 0: {}".format(i, single_model_alphas_by_class[0]))
            for label, class_weight in enumerate(class_weights):
               distances_to_expert_H[i] += class_weight * Hellinger_dist(single_model_alphas_by_class[label], true_alpha_by_class[label] )
               distances_to_expert_C[i] += class_weight * Chernoff_dist(single_model_alphas_by_class[label], true_alpha_by_class[label], args.l)
    else:
        distances_to_expert_H = np.asarray([Hellinger_dist(alpha, true_alpha) for alpha in model_alphas])
        distances_to_expert_C = np.asarray([Chernoff_dist(alpha, true_alpha, args.l) for alpha in model_alphas])


    if args.type == 'datasets':
        labels = DATASIZE_LABELS

    elif args.type == 'models':
        labels = MODEL_LABELS

    elif args.type == 'precise':
        labels = [str(1), str(5), str(10)]
    else:
        raise NotImplementedError(f"Experiment type {args.type} not implemented.")

    # Log the results
    if args.by_class:

        results_dir = oj('{}_results'.format(args.dataset), args.type, 'N{}_byclass-{}'.format(individual_N, args.eval_mode))
        save_results(results_dir, SVs_H, SVs_C, model_alphas_by_class, by_class=True, class_weights=class_weights, mistake_accuracies=eval_accuracies, labels=labels, f1_scores=f1_scores)
        
        np.save(oj(results_dir, 'alphas_by_class.npy'), model_alphas_by_class)
        np.save(oj(results_dir, 'true_alpha_by_class.npy'), true_alpha_by_class)

        np.savetxt(oj(results_dir, 'distances_to_expert_H'), distances_to_expert_H)
        np.savetxt(oj(results_dir, 'distances_to_expert_C'), distances_to_expert_C)

    else:
        results_dir = oj('{}_results'.format(args.dataset), args.type,'N{}-{}'.format(individual_N, args.eval_mode))
        save_results(results_dir, SVs_H, SVs_C, model_alphas, mistake_accuracies=eval_accuracies, labels=labels, f1_scores=f1_scores)
        np.savetxt(oj(results_dir, 'alphas'), model_alphas)
        np.savetxt(oj(results_dir, 'true_alpha'), true_alpha)

        np.savetxt(oj(results_dir, 'distances_to_expert_H'), distances_to_expert_H)
        np.savetxt(oj(results_dir, 'distances_to_expert_C'), distances_to_expert_C)

