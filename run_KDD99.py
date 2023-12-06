
import time
import datetime
from copy import deepcopy

import os
from os.path import join as oj

from KDD99_utils import CNN, CNN_small, LogisticRegression, MLP, KddData

def create_model(method):

    if method == 'CNN':
        original_model = CNN(1, 23)

    elif method == 'CNN_small':
        original_model = CNN_small(1, 23)

    elif method == 'LogisticRegression':
        original_model = LogisticRegression(64, 23)
    else:
        method = 'MLP'
        original_model = MLP(64, 23)

    return original_model


def train_in_parallel(ML_models, optimizers, train_loader, epoch, n_workers=8, device=torch.device('cuda'), schedulers=[]):

    with Pool(n_workers) as pool:

        if len(schedulers) == len(ML_models):
            input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch) for model, optimizer, scheduler in zip(ML_models, optimizers, schedulers) ]
        else:
            input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch) for model, optimizer in zip(ML_models, optimizers) ]

        ML_models = pool.starmap(train, input_arguments)


    return ML_models

def test_in_parallel(ML_models, test_loader, n_workers=8):

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(test_loader)) for model in ML_models ]
        test_accuracies = pool.starmap(test, input_arguments)

    return test_accuracies


from utils_ML import train, test
from utils import get_f1_score

def train_store_models(train_loader, test_loader, num_models=5, method='CNN', n_workers=8, epoch=50):

    original_model = create_model(method)
    
    ML_models = []
    optimizers = []
    for _ in range(num_models):
        model = deepcopy(original_model).to(torch.device('cuda'))
        ML_models.append(model)

        optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
        optimizers.append(optimizer)

    ML_models = train_in_parallel(ML_models, optimizers, train_loader, epoch, n_workers)

    test_accuracies =  test_in_parallel(ML_models, test_loader)
    print("Test accuracies for {}:".format(method), test_accuracies)


    f1_scores = [ get_f1_score(model, torch.device('cuda'), test_loader) for model in ML_models]
    print("F1 scores for {}:".format(method), f1_scores)


    os.makedirs(method, exist_ok=True)
    for i, model in enumerate(ML_models):
        torch.save(model.state_dict(), oj(method, '-saved_model-{}.pt'.format(i+1)))


def train_store_models_datasets(train_loader, test_loader, dataset_proportion=1, num_models=5, method='CNN', n_workers=8, epoch=200):
    
    original_model = create_model(method)

    ML_models = []
    optimizers = []

    for _ in range(num_models):
        model = deepcopy(original_model).to(torch.device('cuda'))
        ML_models.append(model)

        optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
        optimizers.append(optimizer)

    ML_models = train_in_parallel(ML_models, optimizers, train_loader, epoch, n_workers)

    test_accuracies =  test_in_parallel(ML_models, test_loader)
    print("Test accuracies for {}:".format(method), test_accuracies)

    f1_scores = [ get_f1_score(model, torch.device('cuda'), test_loader) for model in ML_models]
    print("F1 scores for {}:".format(method), f1_scores)

    os.makedirs(str(dataset_proportion), exist_ok=True)

    for i, model in enumerate(ML_models):
        torch.save(model.state_dict(), oj(str(dataset_proportion), '-saved_model-{}.pt'.format(i+1)))


import torch
import torch.optim as optim

import os
from os.path import join as oj
import time
import datetime


from multiprocessing.pool import ThreadPool as Pool
from copy import deepcopy

import argparse


from utils import cwd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process which type of training to conduct.')
    parser.add_argument('-N', '--num_models', help='The number of models for a class of model or a type of training.', type=int, default=3)
    parser.add_argument('-t', '--type', help='The type of experiments.', type=str, default='models', choices=['datasets', 'models'])


    args = parser.parse_args()
    print(args)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')


    dataset = KddData(128)

    train_dataset, test_dataset = dataset.train_dataset, dataset.test_dataset
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=1, pin_memory=True)

    if args.type == 'models':

        # using 1% to speed up training
        dataset_proportion = 0.1
        biggest = list(range(0, len(train_dataset), int(1//dataset_proportion)))
        trainset_1 = torch.utils.data.Subset(train_dataset, biggest)
        train_loader = torch.utils.data.DataLoader(trainset_1, batch_size=128, num_workers=1, pin_memory=True)

        exp_dir = oj('saved_models', 'KDDCup', 'models_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='CNN', epoch=50)
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='LogisticRegression', epoch=50)
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='MLP', epoch=50)

    elif args.type == 'datasets':

        exp_dir = oj('saved_models', 'KDDCup', 'datasets_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):

            dataset_proportion = 0.001
            smallest = list(range(0, len(train_dataset), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(train_dataset, smallest)
            train_loader_smallest = torch.utils.data.DataLoader(trainset_1, batch_size=128, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smallest), len(train_loader_smallest), dataset_proportion))
            train_store_models_datasets(train_loader_smallest, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='LogisticRegression', epoch=30)

            dataset_proportion = 0.01
            smaller = list(range(0, len(train_dataset), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(train_dataset, smaller)
            train_loader_smaller = torch.utils.data.DataLoader(trainset_1, batch_size=128, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smaller), len(train_loader_smaller), dataset_proportion))
            train_store_models_datasets(train_loader_smaller, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='LogisticRegression', epoch=30)


            dataset_proportion = 0.1
            biggest = list(range(0, len(train_dataset), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(train_dataset, biggest)
            print('Length of dataset {} for proporation {}'.format(len(trainset_1), dataset_proportion))
            train_loader = torch.utils.data.DataLoader(trainset_1, batch_size=128, num_workers=1, pin_memory=True)
            train_store_models_datasets(train_loader, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='LogisticRegression', epoch=30)