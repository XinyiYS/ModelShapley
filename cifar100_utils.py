import os
import torch
from torchvision import datasets, transforms


def get_loaders():

    # CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    # CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    stats = ((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    # Data transforms (normalization & data augmentation)
    # stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                             transforms.RandomHorizontalFlip(), 
                             transforms.ToTensor(), 
                             transforms.Normalize(*stats,inplace=True)
                            ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)
                            ])
    batch_size = 64
    shuffle = True
    trainset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)

    # tens = list(range(0, len(trainset), 10))
    # sub_trainset = torch.utils.data.Subset(trainset, tens)

    train_loader = torch.utils.data.DataLoader(trainset, shuffle=shuffle, num_workers=4, batch_size=batch_size)


    testset = datasets.CIFAR100(root='data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, shuffle=shuffle, num_workers=4, batch_size=256)


    return train_loader, test_loader
