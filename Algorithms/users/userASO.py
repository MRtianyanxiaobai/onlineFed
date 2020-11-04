import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from Algorithms.optimizers.optimizer import ASOOptimizer
from Algorithms.users.userBase import User

class UserASO(User):
    def __init__(self, id, train_data, test_data, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load):
        super().__init__(id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load)
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        self.optimizer = ASOOptimizer(self.model, lr=self.learning_rate, lamda=self.lamda, beta=self.beta)
    
    def train(self, new_data_num, server):
        LOSS = 0
        self.model.train()
        self.update_data_loader(new_data_num)
        global_model = self.get_global_parameters(server)
        if self.optimizer_method == 'SGD':
            for p, new_param in zip(self.model.parameters(), global_model):
                p.data = new_param.clone()
        for epoch in range(1, self.local_epochs+1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(global_model)
        
        server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)

        return LOSS


