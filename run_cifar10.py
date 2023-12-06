
import os
from os.path import join as oj
import time
import datetime

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

from utils_ML import train, test

def train_store_models(train_loader, test_loader, num_models=5, method='densenet', n_workers=8, epoch=50):

    if method == 'resnet':
        # model_fcn = models.resnet18
        original_model = models.resnet18(pretrained=True)
        original_model.fc = nn.Linear(512, 10)

    elif method == 'squeezenet':
        original_model = models.squeezenet1_0(pretrained=True)
        original_model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1))


    elif method == 'densenet':
        original_model = models.densenet121(pretrained=True)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = original_model.classifier.in_features
        original_model.classifier = nn.Linear(num_ftrs, 10)
        # input_size = 224

    elif method == 'vgg11':
        original_model = models.vgg11_bn(pretrained=True)
        # set_parameter_requires_grad(original_model, feature_extract)
        num_ftrs = original_model.classifier[6].in_features
        original_model.classifier[6] = nn.Linear(num_ftrs, 10)

    else:
        # model_fcn = models.vgg16
        method = 'vgg'
        original_model = models.vgg16(pretrained=True)
        original_model.classifier[6] = nn.Linear(4096, 10)


    ML_models = []
    optimizers = []
    schedulers = []

    for _ in range(num_models):
        model = deepcopy(original_model).to(torch.device('cuda'))
        ML_models.append(model)

        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        optimizers.append(optimizer)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        schedulers.append(scheduler)


    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch, scheduler) for model, optimizer, scheduler in zip(ML_models, optimizers, schedulers) ]
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

def train_store_models_datasets(train_loader, test_loader, dataset_proportion=1, num_models=5, method='densenet', n_workers=8, epoch=100):

    if method == 'resnet':
        # model_fcn = models.resnet18
        original_model = models.resnet18(pretrained=True)
        original_model.fc = nn.Linear(512, 10)

    elif method == 'squeezenet':
        original_model = models.squeezenet1_0(pretrained=True)
        original_model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1))


    elif method == 'densenet':
        original_model = models.densenet121(pretrained=True)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = original_model.classifier.in_features
        original_model.classifier = nn.Linear(num_ftrs, 10)
        # input_size = 224

    ML_models = []
    optimizers = []
    schedulers = []

    for _ in range(num_models):
        model = deepcopy(original_model).to(torch.device('cuda'))
        ML_models.append(model)

        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        optimizers.append(optimizer)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        schedulers.append(scheduler)

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch, scheduler) for model, optimizer, scheduler in zip(ML_models, optimizers, schedulers) ]
        output = pool.starmap(train, input_arguments)

    ML_models = output

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(test_loader)) for model in ML_models ]
        output = pool.starmap(test, input_arguments)
    print("Test accuracies for {}:".format(method), output)

    os.makedirs(str(dataset_proportion), exist_ok=True)

    for i,model in enumerate(ML_models):
        torch.save(model.state_dict(), oj(str(dataset_proportion), '-saved_model-{}.pt'.format(i+1)))

    return

def load_model(method='resnet'):

    if method == 'resnet':
        # model_fcn = models.resnet18
        original_model = models.resnet18(pretrained=True)
        original_model.fc = nn.Linear(512, 10)

    elif method == 'squeezenet':
        original_model = models.squeezenet1_0(pretrained=True)
        original_model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1))

    else:
        # model_fcn = models.vgg16
        method = 'vgg'
        original_model = models.vgg16(pretrained=True)
        original_model.classifier[6] = nn.Linear(4096, 10)


    ML_models = []

    for _ in range(5):
        model = deepcopy(original_model).to(torch.device('cuda'))
        ML_models.append(model)

    return

    
from torchvision import datasets, transforms
import os
from os.path import join as oj
import time
import datetime

import argparse
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool


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
    # test_kwargs = {'batch_size': 512}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)
    
    tens = list(range(0, len(trainset), 10))
    sub_trainset = torch.utils.data.Subset(trainset, tens)

    train_loader = torch.utils.data.DataLoader(
        sub_trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2)


    if args.type == 'models':

        train_loader = torch.utils.data.DataLoader(sub_trainset, **train_kwargs,num_workers=1, pin_memory=True)

        exp_dir = oj('saved_models', 'CIFAR-10', 'models_variation', st)

        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='resnet', epoch=100)
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='squeezenet', epoch=100)
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='densenet', epoch=100)

    elif args.type == 'datasets':

        exp_dir = oj('saved_models', 'CIFAR-10', 'datasets_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):
            dataset_proportion = 0.01
            smallest = list(range(0, len(trainset), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(trainset, smallest)
            train_loader_smallest = torch.utils.data.DataLoader(trainset_1, **train_kwargs, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smallest), len(train_loader_smallest), dataset_proportion))
            train_store_models_datasets(train_loader_smallest, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='resnet', epoch=100)

            dataset_proportion = 0.1
            smaller = list(range(0, len(trainset), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(trainset, smaller)
            train_loader_smaller = torch.utils.data.DataLoader(trainset_1, **train_kwargs, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smaller), len(train_loader_smaller), dataset_proportion))
            train_store_models_datasets(train_loader_smaller, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='resnet', epoch=100)

            dataset_proportion = 1
            print('Length of dataset {} for proporation {}'.format(len(trainset), dataset_proportion))
            train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs, num_workers=1, pin_memory=True)
            train_store_models_datasets(train_loader, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='resnet', epoch=100)
