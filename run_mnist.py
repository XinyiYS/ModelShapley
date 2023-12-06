import os
from os.path import join as oj
import time
import datetime

from multiprocessing.pool import ThreadPool as Pool
from copy import deepcopy

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import ZeroPad2d


from mnist_utils import CNN_Net, MNIST_LogisticRegression, MLP_Net
from utils_ML import train, test

def train_store_models(train_loader, test_loader, num_models=5, method='CNN', n_workers=8, epoch=200):
    '''
    Training models of different model types.
    
    '''

    if method == 'CNN':
        model_fcn = CNN_Net
    elif method == 'MLP':
        model_fcn = MLP_Net
    else:
        model_fcn = MNIST_LogisticRegression
        method = 'LR'

    ML_models = []
    optimizers = []

    for _ in range(num_models):
        model = model_fcn()
        optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
        ML_models.append(model)
        optimizers.append(optimizer)

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch) for model, optimizer in zip(ML_models, optimizers) ]
        output = pool.starmap(train, input_arguments)

    ML_models = output

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(test_loader)) for model in ML_models ]
        output = pool.starmap(test, input_arguments)
    print("Test accuracies for {}:".format(method), output)

    os.makedirs(method, exist_ok=True)

    for i,model in enumerate(ML_models):
        torch.save(model.state_dict(), oj(method, '-saved_model-{}.pt'.format(i+1)))
    return


def train_store_models_datasets(train_loader, test_loader, dataset_proportion=1, num_models=5, method='CNN', n_workers=8, epoch=200):
    '''
    Train models with different sizes of training data.
    
    '''

    if method == 'CNN':
        model_fcn = CNN_Net
    elif method == 'MLP':
        model_fcn = MLP_Net
    else:
        model_fcn = MNIST_LogisticRegression
        method = 'LR'

    ML_models = []
    optimizers = []

    for _ in range(num_models):
        model = model_fcn()
        optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
        ML_models.append(model)
        optimizers.append(optimizer)

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch) for model, optimizer in zip(ML_models, optimizers) ]
        output = pool.starmap(train, input_arguments)

    ML_models = output

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(test_loader)) for model in ML_models ]
        output = pool.starmap(test, input_arguments)
    print("Test accuracies for {}:".format(method), output)

    '''
    with Pool(n_workers) as pool:
        input_arguments = [(model, deepcopy(test_loader)) for model in ML_models]
        model_alphas = pool.starmap(get_mle, input_arguments)
    print("Model alphas complete.", model_alphas[0].shape)
    '''
    os.makedirs(str(dataset_proportion), exist_ok=True)

    for i,model in enumerate(ML_models):
        torch.save(model.state_dict(), oj(str(dataset_proportion), '-saved_model-{}.pt'.format(i+1)))
    return

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

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 512}

    transform=transforms.Compose([
        transforms.ToTensor(),    
        ZeroPad2d(2),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                        transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs, num_workers=1, pin_memory=True)

    tens = list(range(0, len(dataset1), 10))
    trainset_1 = torch.utils.data.Subset(dataset1, tens)

    if args.type == 'models':

        train_loader = torch.utils.data.DataLoader(trainset_1, **train_kwargs, num_workers=1, pin_memory=True)

        exp_dir = oj('saved_models', 'MNIST', 'models_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='CNN')
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='MLP')
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='LR')

    elif args.type == 'datasets':

        exp_dir = oj('saved_models', 'MNIST', 'datasets_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):
            dataset_proportion = 0.01
            smallest = list(range(0, len(dataset1), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(dataset1, smallest)
            train_loader_smallest = torch.utils.data.DataLoader(trainset_1, **train_kwargs, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smallest), len(train_loader_smallest), dataset_proportion))
            train_store_models_datasets(train_loader_smallest, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='CNN', epoch=100)

            dataset_proportion = 0.1
            smaller = list(range(0, len(dataset1), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(dataset1, smaller)
            train_loader_smaller = torch.utils.data.DataLoader(trainset_1, **train_kwargs, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smaller), len(train_loader_smaller), dataset_proportion))
            train_store_models_datasets(train_loader_smaller, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='CNN', epoch=100)

            dataset_proportion = 1
            print('Length of dataset {} for proporation {}'.format(len(dataset1), dataset_proportion))
            train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, num_workers=1, pin_memory=True)
            train_store_models_datasets(train_loader, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='CNN', epoch=100)