import numpy as np
import os
from os.path import join as oj
import time
import datetime
from copy import deepcopy

# %matplotlib inline
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as om
import torchvision as tv
from torchvision import models


from MedNIST_utils import MedSmallestNet, MedSmallNet, MedNet
from utils_ML import train, test


def train_store_models(train_loader, test_loader, num_models=3, method='mednet', n_workers=8, epoch=30):

    if method == 'resnet':
        # resnet18 for normal runs
        resnet = models.resnet18(pretrained=True)
        # change first layer
        resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change last layer
        fc_in = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_in, 6)
        original_model = resnet

    elif method == 'smallnet':
        original_model = MedSmallNet()

    elif method == 'smallestnet' :
        original_model = MedSmallestNet()
    else:
        method = 'mednet'
        original_model = MedNet()

    ML_models = []
    optimizers = []

    for _ in range(num_models):
        model = deepcopy(original_model).to(torch.device('cuda'))
        optimizer = om.SGD(model.parameters(), lr=1e-2, weight_decay=5e-5)
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


def train_store_models_datasets(train_loader, test_loader, dataset_proportion=1, num_models=5, method='mednet', n_workers=8, epoch=30):

    if method == 'resnet':
        # resnet18 for normal runs
        resnet = models.resnet18(pretrained=True)
        # change first layer
        resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change last layer
        fc_in = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_in, 6)
        original_model = resnet

    elif method == 'smallnet':
        original_model = MedSmallNet()

    elif method == 'smallestnet' :
        original_model = MedSmallestNet()
    else:
        method = 'mednet'
        original_model = MedNet()

    ML_models = []
    optimizers = []

    for _ in range(num_models):
        model = deepcopy(original_model).to(torch.device('cuda'))
        optimizer = om.SGD(model.parameters(), lr=1e-2, weight_decay=5e-5)
        ML_models.append(model)
        optimizers.append(optimizer)

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch, None, True) for model, optimizer in zip(ML_models, optimizers) ]
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



import os
from os.path import join as oj
import argparse
from multiprocessing.pool import ThreadPool as Pool


import numpy as np

from utils import cwd

def get_train_test_datasets():

    np.random.seed(551)

    dataDir = oj('data', 'MedNIST')               # The main data directory
    classNames = os.listdir(dataDir)  # Each type of image can be found in its own subdirectory
    numClass = len(classNames)        # Number of types = number of subdirectories
    imageFiles = [[os.path.join(dataDir,classNames[i],x) for x in os.listdir(os.path.join(dataDir,classNames[i]))]
                  for i in range(numClass)]                     # A nested list of filenames
    numEach = [len(imageFiles[i]) for i in range(numClass)]     # A count of each type of image
    imageFilesList = []               # Created an un-nested list of filenames
    imageClass = []                   # The labels -- the type of each individual image in the list
    for i in range(numClass):
        imageFilesList.extend(imageFiles[i])
        imageClass.extend([i]*numEach[i])
    numTotal = len(imageClass)        # Total number of images
    imageWidth, imageHeight = Image.open(imageFilesList[0]).size         # The dimensions of each image

    print("There are",numTotal,"images in",numClass,"distinct categories")
    print("Label names:",classNames)
    print("Label counts:",numEach)
    print("Image dimensions:",imageWidth,"x",imageHeight)

    toTensor = tv.transforms.ToTensor()
    def scaleImage(x):          # Pass a PIL image, return a tensor
        y = toTensor(x)
        if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
            y = (y - y.min())/(y.max() - y.min()) 
        z = y - y.mean()        # Subtract the mean value of the image
        return z


    imageTensor = torch.stack([scaleImage(Image.open(x)) for x in imageFilesList])  # Load, scale, and stack image (X) tensor
    classTensor = torch.tensor(imageClass)  # Create label (Y) tensor


    validFrac = 0   # Define the fraction of images to move to validation dataset
    testFrac = 0.2    # Define the fraction of images to move to test dataset
    validList = []
    testList = []
    trainList = []

    for i in range(numTotal):
        rann = np.random.random() # Randomly reassign images
        if rann < validFrac:
            validList.append(i)
        elif rann < testFrac + validFrac:
            testList.append(i)
        else:
            trainList.append(i)
            
    nTrain = len(trainList)  # Count the number in each set
    nValid = len(validList)
    nTest = len(testList)
    print("Training images =",nTrain,"Validation =",nValid,"Testing =",nTest)

    trainIds = torch.tensor(trainList)    # Slice the big image and label tensors up into
    # validIds = torch.tensor(validList)    #       training, validation, and testing tensors
    testIds = torch.tensor(testList)
    trainX = imageTensor[trainIds,:,:,:]
    trainY = classTensor[trainIds]
    # validX = imageTensor[validIds,:,:,:]
    # validY = classTensor[validIds]
    testX = imageTensor[testIds,:,:,:]
    testY = classTensor[testIds]

    # Create Torch datasets
    train_set = torch.utils.data.TensorDataset(trainX, trainY)
    # valid_set = torch.utils.data.TensorDataset(validX, validY)
    test_set = torch.utils.data.TensorDataset(testX, testY)

    return train_set, test_set


if __name__ == '__main__':


    np.random.seed(551)

    parser = argparse.ArgumentParser(description='Process which type of training to conduct.')
    parser.add_argument('-N', '--num_models', help='The number of models for a class of model or a type of training.', type=int, default=3)
    parser.add_argument('-t', '--type', help='The type of experiments.', type=str, default='models', choices=['datasets', 'models'])


    args = parser.parse_args()
    print(args)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')

    train_kwargs = {'batch_size': 300}
    test_kwargs = {'batch_size': 300}

    '''
    learnRate = 0.01          # Define a learning rate.
    E = 30            # Maximum training epochs
    t2vRatio = 1.2            # Maximum allowed ratio of validation to training loss
    t2vEpochs = 3             # Number of consecutive epochs before halting if validation loss exceeds above limit
    batchSize = 300           # Batch size. Going too large will cause an out-of-memory error.
    trainBats = nTrain // batchSize       # Number of training batches per epoch. Round down to simplify last batch
    validBats = nValid // batchSize       # Validation batches. Round down
    testBats = -(-nTest // batchSize)     # Testing batches. Round up to include all
    
    CEweights = torch.zeros(6)     # This takes into account the imbalanced dataset.
    for i in trainY.tolist():             #      By making rarer images count more to the loss, 
        CEweights[i].add_(1)              #      we prevent the model from ignoring them.
    CEweights = 1. / CEweights.clamp_(min=1.)                     # Weights should be inversely related to count
    CEweights = (CEweights * 6 / CEweights.sum()).to(dev)  # The weights average to 1
    '''

    train_set, test_set = get_train_test_datasets()
    print('Lengths of train_set and test_set: ', len(train_set), len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=300, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=300, shuffle=False, pin_memory=True)


    if args.type == 'models':
        exp_dir = oj('saved_models', 'MedNIST', 'models_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='mednet')
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='resnet')
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='smallestnet')

    elif args.type == 'datasets':

        exp_dir = oj('saved_models', 'MedNIST', 'datasets_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):

            dataset_proportion = 0.01
            smallest = list(range(0, len(train_set), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(train_set, smallest)
            train_loader_smallest = torch.utils.data.DataLoader(trainset_1, **train_kwargs, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proportion {}'.format(len(smallest), len(train_loader_smallest), dataset_proportion))
            train_store_models_datasets(train_loader_smallest, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='mednet', epoch=30)

            dataset_proportion = 0.1
            smaller = list(range(0, len(train_set), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(train_set, smaller)
            train_loader_smaller = torch.utils.data.DataLoader(trainset_1, **train_kwargs, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proportion {}'.format(len(smaller), len(train_loader_smaller), dataset_proportion))
            train_store_models_datasets(train_loader_smaller, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='mednet', epoch=30)

            dataset_proportion = 1
            print('Length of dataset {} for proportion {}'.format(len(train_set), dataset_proportion))
            train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs, num_workers=1, pin_memory=True)
            train_store_models_datasets(train_loader, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='mednet', epoch=30)
