import os
from os.path import join as oj

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
import torchvision as tv


class MedSmallestNet(nn.Module):
    def __init__(self, numClass=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1, 1)
        # self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.fc1 = nn.Linear(128, numClass)
        # self.fc2 = nn.Linear(8, numClass)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), 8, 8)
        x = torch.flatten(x, 1)
        return self.fc1(x)


class MedSmallNet(nn.Module):
    def __init__(self, numClass=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.fc1 = nn.Linear(384, 8)
        self.fc2 = nn.Linear(8, numClass)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), 4, 4)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MedNet(nn.Module):
    def __init__(self,xDim=64,yDim=64,numC=6): # Pass image dimensions and number of labels when initializing a model   
        super(MedNet,self).__init__()  # Extends the basic nn.Module to the MedNet class
        # The parameters here define the architecture of the convolutional portion of the CNN. Each image pixel
        # has numConvs convolutions applied to it, and convSize is the number of surrounding pixels included
        # in each convolution. Lastly, the numNodesToFC formula calculates the final, remaining nodes at the last
        # level of convolutions so that this can be "flattened" and fed into the fully connected layers subsequently.
        # Each convolution makes the image a little smaller (convolutions do not, by default, "hang over" the edges
        # of the image), and this makes the effective image dimension decreases.
        
        numConvs1 = 5
        convSize1 = 7
        numConvs2 = 10
        convSize2 = 7
        numNodesToFC = numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1))

        # nn.Conv2d(channels in, channels out, convolution height/width)
        # 1 channel -- grayscale -- feeds into the first convolution. The same number output from one layer must be
        # fed into the next. These variables actually store the weights between layers for the model.
        
        self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
        self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)

        # These parameters define the number of output nodes of each fully connected layer.
        # Each layer must output the same number of nodes as the next layer begins with.
        # The final layer must have output nodes equal to the number of labels used.
        
        fcSize1 = 400
        fcSize2 = 80
        
        # nn.Linear(nodes in, nodes out)
        # Stores the weights between the fully connected layers
        
        self.ful1 = nn.Linear(numNodesToFC,fcSize1)
        self.ful2 = nn.Linear(fcSize1, fcSize2)
        self.ful3 = nn.Linear(fcSize2,numC)
        
    def forward(self,x):
        # This defines the steps used in the computation of output from input.
        # It makes uses of the weights defined in the __init__ method.
        # Each assignment of x here is the result of feeding the input up through one layer.
        # Here we use the activation function elu, which is a smoother version of the popular relu function.
        
        x = F.elu(self.cnv1(x)) # Feed through first convolutional layer, then apply activation
        x = F.elu(self.cnv2(x)) # Feed through second convolutional layer, apply activation
        x = x.view(-1,self.num_flat_features(x)) # Flatten convolutional layer into fully connected layer
        x = F.elu(self.ful1(x)) # Feed through first fully connected layer, apply activation
        x = F.elu(self.ful2(x)) # Feed through second FC layer, apply output
        x = self.ful3(x)        # Final FC layer to output. No activation, because it's used to calculate loss
        return x

    def num_flat_features(self, x):  # Count the individual nodes in a layer
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




# MODEL_LABELS =['Res', 'Sml', 'Tny']
MODEL_LABELS =['Med', 'Res', 'Tny']
DATASIZE_LABELS = [str(0.01), str(0.1), str(1)]

def get_loaders():
    
    # if torch.cuda.is_available():     # Make sure GPU is available
    #     dev = torch.device("cuda:0")
    #     kwar = {'num_workers': 8, 'pin_memory': True}
    #     cpu = torch.device("cpu")
    # else:
    #     print("Warning: CUDA not found, CPU only.")
    #     dev = torch.device("cpu")
    #     kwar = {}
    #     cpu = torch.device("cpu")

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


    validFrac = 0.   # Define the fraction of images to move to validation dataset
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

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=300, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=300, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=300, shuffle=False)

    return train_loader, test_loader


def get_resnet(numClass=6):

    # resnet18 for normal runs
    resnet = models.resnet18(pretrained=False)
    # change first layer
    resnet.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # change last layer
    fc_in = resnet.fc.in_features
    resnet.fc = nn.Linear(fc_in, numClass)
    return resnet


from utils import cwd

def get_models(individual_N=3, exp_type='models'):
    
    '''
    Load saved models. 

    NOTE: Change the directories to your saved models.
    
    '''
    
    if exp_type == 'datasets':
        models = []
        exp_dir = oj('saved_models', 'MedNIST', 'datasets_variation', '2022-01-26-17:18')
        with cwd(exp_dir):
            print("Loading order of dataset proportions:", sorted(os.listdir(), key=float))                  
            for saved_dir in sorted(os.listdir(), key=float):
                for i in range(individual_N):
                    model = MedNet()
                    model.load_state_dict(torch.load(oj(saved_dir,'-saved_model-{}.pt'.format(i+1))))
                    models.append(model)

        return models

    elif exp_type == 'models':

        models = []
        exp_dir = oj('saved_models', 'MedNIST', 'models_variation', '2022-01-26-17:25')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = MedNet()
                model.load_state_dict(torch.load(oj('mednet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = get_resnet()
                model.load_state_dict(torch.load(oj('resnet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = MedSmallestNet()
                model.load_state_dict(torch.load(oj('smallestnet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)
        return models


    elif exp_type == 'precise':        
        models = []
        exp_dir = oj('saved_models', 'MedNIST', 'models_variation', '2022-01-26-17:25')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = MedNet()
                model.load_state_dict(torch.load(oj('mednet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = MedNet()
                model.load_state_dict(torch.load(oj('mednet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)
            for i in range(individual_N):
                model = MedNet()
                model.load_state_dict(torch.load(oj('mednet', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

        return models
    else:
        raise NotImplementedError(f"Experiment type: {exp_type} is not implemented.")

