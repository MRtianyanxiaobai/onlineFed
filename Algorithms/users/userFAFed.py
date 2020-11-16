import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from Algorithms.optimizers.optimizer import ASOOptimizer
from Algorithms.users.userBase import User

class UserFAFed(User):
    def __init__(self, id, train_data, test_data, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load):
        super().__init__(id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load)
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.model_copy = copy.deepcopy(list(model[0].parameters()))

    def train(self, server):
        if self.can_train() == False:
            return False
        else: 
            if self.trained == True:
                server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
                self.trained = False
        
        self.model_copy = copy.deepcopy(list(self.model.parameters()))
        global_model = self.get_global_parameters(server)
        for p, new_param in zip(self.model.parameters(), global_model):
            p.data = new_param.clone()
        self.model.eval()
        updated_stats = self.test()
        updated_acc = updated_stats[0]*1.0/updated_stats[1]
        if updated_acc < self.test_acc:
            for updated_param, old_param in zip(self.model.parameters(), self.model_copy):
                updated_param.data = old_param.data + 0.3*(updated_param.data - old_param.data)
                
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs+1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        
        self.trained = True
        if self.check_async_update():
            server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
            self.trained = False

        return LOSS