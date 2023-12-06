import os
from os.path import join as oj

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision.models as torch_models

MODEL_LABELS =['Res', 'Sqz', 'Den']
DATASIZE_LABELS = [str(0.01), str(0.1), str(1)]


def get_loaders():
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
        testset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader


from utils import cwd

def get_models(individual_N=3, exp_type='models'):

    '''
    Load saved models. 

    NOTE: Change the directories to your saved models.
    
    '''
    if exp_type == 'datasets':
        models = []
        exp_dir = oj('saved_models', 'CIFAR-10', 'datasets_variation', '2022-01-26-09:35')
        with cwd(exp_dir):
            print("Loading order of dataset proportions:", sorted(os.listdir(), key=float))                  
            for saved_dir in sorted(os.listdir(), key=float):
                for i in range(individual_N):
                    model = torch_models.resnet18(pretrained=False)
                    model.fc = nn.Linear(512, 10)
                
                    model.load_state_dict(torch.load(oj(saved_dir,'-saved_model-{}.pt'.format(i+1))))
                    models.append(model)

        return models

    elif exp_type == 'models':

        models = []
        exp_dir = oj('saved_models', 'CIFAR-10', 'models_variation', '2022-01-26-09:36')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = torch_models.resnet18(pretrained=False)
                model.fc = nn.Linear(512, 10)
                
                model.load_state_dict(torch.load(oj('resnet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = torch_models.squeezenet1_0(pretrained=False)
                model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1))
                model.load_state_dict(torch.load(oj('squeezenet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = torch_models.densenet121(pretrained=False)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, 10)
                model.load_state_dict(torch.load(oj('densenet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

        return models

    elif exp_type == 'precise':

        models = []
        exp_dir = oj('saved_models', 'CIFAR-10', 'models_variation', '2022-01-26-09:36')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = torch_models.densenet121(pretrained=False)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, 10)
                model.load_state_dict(torch.load(oj('densenet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = torch_models.densenet121(pretrained=False)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, 10)
                model.load_state_dict(torch.load(oj('densenet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = torch_models.densenet121(pretrained=False)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, 10)
                model.load_state_dict(torch.load(oj('densenet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

        return models
    else:
        raise NotImplementedError(f"Experiment type: {exp_type} is not implemented.")

