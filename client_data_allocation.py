import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.datasets import fetch_20newsgroups
import os
import json
import numpy as np
from scipy import stats
from PIL import Image
import random
from fed_utils import test_batch_cls

seed = 42
np.random.seed(seed)
random.seed(seed)

DATA = "."

def build_dataset(batch_size, n_clients, alpha=-1, seed=0):
    valset = None
    clients, valset, testset = build_cifar10(n_clients, alpha, seed)
    TEST_BATCH = 32
    clientloaders = [DataLoader(client, batch_size=batch_size, shuffle=True, num_workers=0) for client in clients]
    if valset is not None:
        valloader = DataLoader(valset, batch_size=TEST_BATCH, shuffle=False, num_workers=1)
    else:
        valloader = None
    testloader = DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=1)
    def test_batch(model, x, y):
        return test_batch_cls(model, x, y)
    return clientloaders, valloader, testloader, test_batch

def partition_dirichlet(Y, n_clients, alpha, seed):
    clients = []
    ex_per_class = np.unique(Y, return_counts=True)[1]
    n_classes = len(ex_per_class)
    print(f"Found {n_classes} classes")
    rv_tr = stats.dirichlet.rvs(np.repeat(alpha, n_classes), size=n_clients, random_state=seed) 
    rv_tr = rv_tr / rv_tr.sum(axis=0)
    rv_tr = (rv_tr*ex_per_class).round().astype(int)
    class_to_idx = {i: np.where(Y == i)[0] for i in range(n_classes)}
    curr_start = np.zeros(n_classes).astype(int)
    for client_classes in rv_tr:
        curr_end = curr_start + client_classes
        client_idx = np.concatenate([class_to_idx[c][curr_start[c]:curr_end[c]] for c in range(n_classes)])
        curr_start = curr_end
        clients.append(client_idx)
        # will be empty subset if all examples have been exhausted
    return clients

def build_cifar10(n_clients, alpha, seed):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = torchvision.datasets.CIFAR10(root=f"{DATA}/cifar10", train=True, download=True, transform=transform)
    N = len(trainset)

    #Create a subset with 10000 samples
    total_samples = 10000
    num_classes = 10
    samples_per_class = total_samples // num_classes

    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)
    subset_indices = []
    for _, indices in class_indices.items():
        subset_indices.extend(indices[:samples_per_class])
    random.shuffle(subset_indices)

    trainset = torch.utils.data.Subset(trainset, subset_indices)
    subset_targets = np.array([trainset.dataset.targets[i] for i in trainset.indices])  

    trainidx = np.arange(0, int(len(subset_indices) * 0.8))
    Y_tr = subset_targets[trainidx]  # Access targets for the training set
    clientidx = partition_dirichlet(Y_tr, n_clients, alpha, seed)
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]

    validx = np.arange(int(len(subset_indices) * 0.8), len(subset_indices))
    valset = torch.utils.data.Subset(trainset, validx)
    testset = torchvision.datasets.CIFAR10(root=f"{DATA}/cifar10", train=False, download=True, transform=test_transform)
    return clients, valset, testset
