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
        super().__init__(id, train_data, test_data, model[0], async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = lamdaSGDOptimizer(self.model, lr=self.learning_rate, lamda=self.lamda)
        self.last_model = copy.deepcopy(list(model[0].parameters()))
        self.benefit = True

    def train(self, server):
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
                
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs+1):
            # iter_num = int(self.train_data_samples / self.batch_size)
            # for i in range(iter_num):
                self.model.train()
                X, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(global_model)
        
        self.trained = True
        if self.check_async_update():
            server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
            self.trained = False

        return LOSS
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            output = self.model(x.cuda())
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y.cuda())).item()
        last_acc = self.test_acc
        self.test_acc = test_acc*1.0 / y.shape[0]
        if self.test_acc - last_acc >= 0:
            self.benefit = True
        else:
            self.benefit = False
        return test_acc, y.shape[0]