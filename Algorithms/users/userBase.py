import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User:
    def __init__(self, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0, lamda = 0, local_epochs = 0, optimizer = "SGD", data_load = "fixed"):
        self.model = copy.deepcopy(model)
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.beta = beta
        self.local_epochs = local_epochs
        self.optimizer_method = optimizer
        self.data_load = data_load

        self.trained = False
        self.async_delay = 0.6

        self.test_acc = 0

        self.train_data_len = len(self.train_data)
        self.test_data_len = len(self.test_data)
        self.train_data_samples = self.train_data_len if data_load == "fixed" else int(self.train_data_len * 0.5)
        self.test_data_samples = self.test_data_len if data_load == "fixed" else int(self.test_data_len * 0.5)

        self.update_data_loader(0)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

    def update_data_loader(self, new_data):
        train_samples = self.train_data_samples + new_data
        test_samples = self.test_data_samples + new_data
        if train_samples < self.train_data_len:
            self.train_data_samples = train_samples
            self.trainloader = DataLoader(self.train_data[:self.train_data_samples], self.batch_size)
            self.trainloaderfull = DataLoader(self.train_data[:self.train_data_samples], self.train_data_samples)
        else:
            self.train_data_samples = self.train_data_len
            self.trainloader = DataLoader(self.train_data[:self.train_data_samples], self.batch_size)
            self.trainloaderfull = DataLoader(self.train_data[:self.train_data_samples], self.train_data_samples)
        if test_samples < self.test_data_len:
            self.test_data_samples = test_samples
            self.testloader = DataLoader(self.test_data[:self.test_data_samples], self.batch_size)
            self.testloaderfull = DataLoader(self.test_data[:self.test_data_samples], self.test_data_samples)
        else:
            self.test_data_samples = self.test_data_len
            self.testloader = DataLoader(self.test_data[:self.test_data_samples], self.batch_size)
            self.testloaderfull = DataLoader(self.test_data[:self.test_data_samples], self.test_data_samples)
    
    def get_next_train_batch(self):
        try:
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.cuda(), y.cuda())
    
    def get_next_test_batch(self):
        try:
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.cuda(), y.cuda())
    
    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads
    
    def get_local_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def get_global_parameters(self, server):
        return server.model.parameters()
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def has_new_data(self):
        if self.data_load == 'fixed': 
            return False
        data_flag = torch.rand(1).item()
        if data_flag < 0.2:
            self.update_data_loader(int(data_flag*20))
            return True
        return False
    
    def can_train(self):
        if self.trained == False:
            return self.has_new_data()
        else:
            return self.check_async_update()

    def check_async_update(self):
        update_flag = torch.rand(1).item()
        if update_flag < self.async_delay:
            self.async_delay = 0.4
            return True
        else:
            self.async_delay = self.async_delay + 0.1
            return False
            
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            output = self.model(x.cuda())
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y.cuda())).item()
        self.test_acc = test_acc*1.0 / y.shape[0]
        return test_acc, y.shape[0]
    
    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            output = self.model(x.cuda())
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y.cuda())).item()
            loss += self.loss(output, y.cuda())
        return train_acc, loss.item() , self.train_data_samples
