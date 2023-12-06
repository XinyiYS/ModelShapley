import os
from os.path import join as oj

import torch
from torchvision import datasets, transforms
from torch.nn import ZeroPad2d

from utils_ML import test

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Model architectures
# for MNIST 32*32
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 16, 7, 1)
        self.fc1 = nn.Linear(4 * 4 * 16, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)

# for MNIST 32*32 LogReg
class MNIST_LogisticRegression(nn.Module):

    def __init__(self, input_dim=1024, output_dim=10):
        super(MNIST_LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = x.view(-1,  1024)
        outputs = self.linear(x)
        return outputs
        # return F.log_softmax(outputs, dim=1)

# for MNIST 32*32
class MLP_Net(nn.Module):

    def __init__(self):
        super(MLP_Net, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1,  1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # return F.log_softmax(x, dim=1)


def get_loaders():
    train_kwargs = {'batch_size': 32}
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

    tens = list(range(0, len(dataset1), 10))
    trainset_1 = torch.utils.data.Subset(dataset1, tens)

    train_loader = torch.utils.data.DataLoader(trainset_1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader


from utils import cwd

def get_models(individual_N=3, exp_type='models'):

    '''
    Load saved models. 

    NOTE: Change the directories to your saved models.
    
    '''

    if exp_type == 'datasets':
        models = []
        exp_dir = oj('saved_models', 'MNIST', 'datasets_variation', '2022-01-17-15:11')
        with cwd(exp_dir):
            print("Loading order of dataset proportions:", sorted(os.listdir(), key=float))                  
            for saved_dir in sorted(os.listdir(), key=float):
                for i in range(individual_N):
                    model = CNN_Net()
                    model.load_state_dict(torch.load(oj(saved_dir,'-saved_model-{}.pt'.format(i+1))))
                    models.append(model)
        return models

    
    elif exp_type == 'models':

        models = []
        exp_dir = oj('saved_models', 'MNIST', 'models_variation', '2023-12-06-09:46')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = CNN_Net()
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = MLP_Net()
                model.load_state_dict(torch.load(oj('MLP', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = MNIST_LogisticRegression()
                model.load_state_dict(torch.load(oj('LR', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)
        return models

    elif exp_type == 'precise':        
        models = []
        exp_dir = oj('saved_models', 'MNIST', 'models_variation', '2022-01-16-15:55')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = CNN_Net()
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = CNN_Net()
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = CNN_Net()
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

        return models
    else:
        raise NotImplementedError(f"Experiment type: {exp_type} is not implemented.")

MODEL_LABELS =['CNN', 'MLP', 'LR']
DATASIZE_LABELS = [str(0.01), str(0.1), str(1)]



