import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from Algorithms.optimizers.optimizer import lamdaSGDOptimizer
from Algorithms.users.userBase import User

class UserFAFed(User):
    def __init__(self, id, train_data, test_data, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load):
        super().__init__(id, train_data, test_data, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = lamdaSGDOptimizer(self.model, lr=self.learning_rate, lamda=self.lamda)
        self.last_model = copy.deepcopy(list(model.parameters()))
        self.benefit = True

    def run(self, server, glob_iter):
        if self.can_train() == False:
            return False
        else: 
            if self.trained == True:
                server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
                self.trained = False
        
        global_model = self.get_global_parameters(server)
        for global_param, local_param, last_local_param in zip(global_model, self.model.parameters(), self.last_model):
            distance = global_param.data - local_param.data
            local_distance = local_param.data - last_local_param.data
            distance_vec = torch.flatten(distance)
            local_distance_vec = torch.flatten(local_distance)
            similarity = torch.cosine_similarity(local_distance_vec, distance_vec, dim=0).item()
            if similarity >= 0:
                local_param.data = local_param.data + distance
            else:
                local_param.data = local_param.data + self.beta*distance
            last_local_param.data = local_param.data.clone()
                
        self.train(global_model)
        if self.check_async_update():
            server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
            self.trained = False

        return LOSS
    def train(self, global_model):
        LOSS = 0
        # loss_log = []
        self.model.train()
        for epoch in range(1, self.local_epochs+1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            # loss_log.append(loss.item())
            loss.backward()
            self.optimizer.step(global_model)
        # self.loss_log.append(loss_log)
        self.trained = True
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for i, (x, y) in enumerate(self.testloader):
            output = self.model(x.cuda())
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y.cuda())).item()
        last_acc = self.test_acc
        self.test_acc = test_acc*1.0 / self.test_data_samples
        self.test_acc_log.append(self.test_acc)
        if self.test_acc - last_acc >= 0:
            self.benefit = True
        else:
            self.benefit = False
        return test_acc, self.test_data_samples