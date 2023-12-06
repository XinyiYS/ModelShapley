import os
from os.path import join as oj
from math import factorial as fac

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from dirichlet import mle

from utils import cwd


class KddData(object):

    def __init__(self, batch_size):
        kddcup99 = sk_datasets.fetch_kddcup99()
        self._encoder = {
            'protocal': LabelEncoder(),
            'service':  LabelEncoder(),
            'flag':     LabelEncoder(),
            'label':    LabelEncoder()
        }
        self.batch_size = batch_size
        data_X, data_y = self.__encode_data(kddcup99.data, kddcup99.target)
        self.train_dataset, self.test_dataset = self.__split_data_to_tensor(data_X, data_y)
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=True)


    def __encode_data(self, data_X, data_y):
        self._encoder['protocal'].fit(list(set(data_X[:, 1])))
        self._encoder['service'].fit(list(set(data_X[:, 2])))
        self._encoder['flag'].fit((list(set(data_X[:, 3]))))
        self._encoder['label'].fit(list(set(data_y)))
        data_X[:, 1] = self._encoder['protocal'].transform(data_X[:, 1])
        data_X[:, 2] = self._encoder['service'].transform(data_X[:, 2])
        data_X[:, 3] = self._encoder['flag'].transform(data_X[:, 3])
        data_X = np.pad(data_X, ((0, 0), (0, 64 - len(data_X[0]))), 'constant').reshape(-1, 1, 8, 8)
        data_y = self._encoder['label'].transform(data_y)
        return data_X, data_y

    def __split_data_to_tensor(self, data_X, data_y):
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)

        train_dataset = TensorDataset(
            (torch.from_numpy(X_train.astype(float))).float(),
            (torch.from_numpy(y_train)).long()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test.astype(float)).float(),
            torch.from_numpy(y_test).long()
        )
        return train_dataset, test_dataset

    def decode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].inverse_transform([_data[1]])[0]
            _data[2] = self._encoder['service'].inverse_transform([_data[2]])[0]
            _data[2] = self._encoder['flag'].inverse_transform([_data[3]])[0]
            return _data
        return self._encoder['label'].inverse_transform(data)
    
    def encode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].transform([_data[1]])[0]
            _data[2] = self._encoder['service'].transform([_data[2]])[0]
            _data[3] = self._encoder['flag'].transform([_data[3]])[0]
            return _data
        return self._encoder['label'].transform([data])[0]

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(144, 512),
            nn.Linear(512, 256),
            nn.Linear(256, n_class)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNN_small(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN_small, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, n_class)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim=64, output_dim=23, device=None):
        super(MLP, self).__init__()
        self.input_dim=input_dim
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(-1,  self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return x
        return torch.softmax(x, dim=1)

class LogisticRegression(nn.Module):

    def __init__(self, input_dim=64, output_dim=23, device=None):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = x.view(-1,  64)
        outputs = self.linear(x)
        return outputs
        # return torch.softmax(outputs, dim=1)

def get_loaders():

    dataset = KddData(batch_size=128)
    train_dataset, test_dataset = dataset.train_dataset, dataset.test_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=1, pin_memory=True)


    '''
    # subsample it by 1000
    dataset_proportion = 0.01
    smaller = list(range(0, len(test_dataset), int(1//dataset_proportion)))
    
    subsampled_test_set = torch.utils.data.Subset(test_dataset, smaller)
    test_loader = torch.utils.data.DataLoader(subsampled_test_set, batch_size=512, num_workers=1, pin_memory=True)
    '''

    return train_loader, test_loader


def get_test_loader_alpha():
    try:
        true_alpha = np.loadtxt(oj('saved_models', 'KDDCup', 'test_loader_alpha'))
    except:

        _, test_loader =  get_loaders()
        temp = []
        for data, labels in test_loader:
            a = F.one_hot(labels, num_classes=23).float()

            # softmax to get the true alpha
            a = torch.softmax(a, dim=1)
            
            temp.append(a.detach().cpu())

        temp = torch.vstack(temp).numpy()
        true_alpha = mle(temp, method='fixpoint')
        np.savetxt(oj('saved_models', 'KDDCup', 'test_loader_alpha'), np.asarray(true_alpha))

        '''
        X, Y = get_X_Y()
        # Split test and train data 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        enc = OneHotEncoder()
        D = enc.fit_transform(Y_test).toarray()
        # softmax to get the true alpha
        D = softmax(10 * D)

        true_alpha = mle(D)
        np.savetxt(oj('saved_models', 'KDDCup', 'test_loader_alpha'), np.asarray(true_alpha))
        '''

    print('true alpha:', true_alpha)
    return true_alpha


def get_model_alphas(individual_N=3, model_dirs=[]):

    return [np.loadtxt(oj('saved_models', 'KDDCup', model_dir, 'model_alphas'))[:individual_N] for model_dir in model_dirs]


MODEL_LABELS =['CNN', 'MLP', 'LR']
DATASIZE_LABELS = [str(0.001), str(0.01), str(0.1)]

def get_models(individual_N=3, exp_type='models', model_type='LogisticRegression'):

    '''
    Load saved models. 

    NOTE: Change the directories to your saved models.
    
    '''

    if exp_type == 'datasets':
        models = []

        # model_type = 'CNN_small'

        exp_dir = oj('saved_models', 'KDDCup', 'datasets_variation', model_type)
        # exp_dir = oj('saved_models', 'KDDCup', 'datasets_variation', model_type)
        with cwd(exp_dir):
            print("Loading order of dataset proportions:", sorted(os.listdir(), key=float))                  
            for saved_dir in sorted(os.listdir(), key=float):
                for i in range(individual_N):
                    
                    if model_type == 'LogisticRegression':
                        model = LogisticRegression()
                    else:
                        model = CNN_small()
                    
                    model.load_state_dict(torch.load(oj(saved_dir,'-saved_model-{}.pt'.format(i+1))))
                    models.append(model)
        return models

    
    elif exp_type == 'models':

        models = []
        exp_dir = oj('saved_models', 'KDDCup', 'models_variation', '2022-01-26-14:00')
        exp_dir = oj('saved_models', 'KDDCup', 'models_variation', '2022-01-26-14:08')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = CNN(1, 23)
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = MLP(64, 23)
                model.load_state_dict(torch.load(oj('MLP', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = LogisticRegression(64, 23)
                model.load_state_dict(torch.load(oj('LogisticRegression', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)
        return models

    elif exp_type == 'precise':
        models = []
        exp_dir = oj('saved_models', 'KDDCup', 'models_variation', '2022-01-26-14:08')
        exp_dir = oj('saved_models', 'KDDCup', 'models_variation', '2022-01-26-14:00')

        
        with cwd(exp_dir):
            for i in range(individual_N):
                model = CNN(1, 23)
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = CNN(1, 23)
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)
            
            for i in range(individual_N):
                model = CNN(1, 23)
                model.load_state_dict(torch.load(oj('CNN', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)
    
        return models
    else:
        raise NotImplementedError(f"Experiment type: {exp_type} is not implemented.")
