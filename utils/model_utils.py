import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import random
from itertools import combinations
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

def read_data(dataset, niid, num_users, user_labels):
    if niid == False:
        user_labels = 10
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == 'Cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])
        trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
    if dataset == 'FasionMNIST':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    data_value = []
    data_label = []
    data_value.extend(trainset.data.cpu().detach().numpy())
    data_value.extend(testset.data.cpu().detach().numpy())
    data_label.extend(trainset.targets.cpu().detach().numpy())
    data_label.extend(testset.targets.cpu().detach().numpy())
    data_value = np.array(data_value)
    data_label = np.array(data_label)

    data_source = []
    for i in trange(10):
        idx = data_label==i
        data_source.append(data_value[idx])

    if niid == True:
        label_split = [c for c in combinations(range(10), user_labels)]
        random.shuffle(label_split)
        label_split = label_split[:num_users]
    else: 
        label_split = [range(10) for _ in range(num_users)]
    unique, counts = np.unique(label_split, return_counts=True)
    print("--------------")
    print(unique, counts)

    def ram_dom_gen(total, size):
        print(total)
        nums = []
        temp = []
        for i in range(size - 1):
            val = np.random.randint(total//(size - i + 1), total//(size - i - 1))
            temp.append(val)
            total -= val
        temp.append(total)
        print(temp)
        return temp

    number_sample = []
    for total_value, count in zip(data_source, counts):
        temp = ram_dom_gen(len(total_value), count)
        number_sample.append(temp)
    print("--------------")
    print(number_sample)

    X = [[] for _ in range(num_users)]
    y = [[] for _ in range(num_users)]
    idx = np.zeros(shape=10, dtype=np.int)
    part_index = np.zeros(shape=10, dtype=np.int)
    for user in trange(num_users):
        for l in label_split[user]:
            num_samples = number_sample[l][part_index[l]]
            part_index[l] = part_index[l] + 1
            if idx[l] + num_samples <= len(data_source[l]):
                X[user] += data_source[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                print("check len os user, label:", user, l, "len data", len(X[user]), num_samples)
    print("class samples:", idx)

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    for i in range(num_users):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    print("Num_samples:", train_data['num_samples'])
    print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    return train_data['users'], _ , train_data['user_data'], test_data['user_data']

def read_user_data(index,data,dataset):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "Mnist"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar10"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

class Object:
    def __init__(self, data):
        self.__dict__.update(data)