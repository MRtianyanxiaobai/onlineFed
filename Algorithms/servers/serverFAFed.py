import torch
import os
import copy
from Algorithms.servers.serverBase import Server
import numpy as np

class ServerFAFed(Server):
    def __init__(self, algorithm, model, async_process, test_data):
        super().__init__(algorithm, model, async_process, test_data)
        self.last_model = copy.deepcopy(list(model.parameters()))
        self.benefit = True
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        last_acc = self.test_acc
        self.test_acc = test_acc*1.0 / y.shape[0]
        if self.test_acc - last_acc >= 0:
            self.benefit = True
        else:
            self.benefit = False
        return test_acc, y.shape[0]
    
    def aggregate_parameters(self, user_data):
        if self.async_process == True:
            for user_updated in user_data:
                self.users[user_updated.id].samples = user_updated.samples
                total_train = 0
                for user in self.users.values():
                    total_train += user.samples
                for global_param, last_global_param, user_new_param, user_old_param in zip(self.model.parameters(), self.last_model, user_updated.model, self.users[user_updated.id].model):
                    distance = user_new_param.data - global_param.data
                    global_distance = global_param.data - last_global_param.data
                    distance_vec = torch.flatten(distance)
                    global_distance_vec = torch.flatten(global_distance)
                    similarity = torch.cosine_similarity(global_distance_vec, distance_vec, dim=0).item()
                    last_global_param.data = global_param.data.clone()
                    if self.benefit == True and similarity >= 0:
                        global_param.data = global_param.data + 1.1*(user_updated.samples / total_train)*(user_new_param.data - user_old_param.data)
                    elif self.benefit == False and similarity <= 0:
                        global_param.data = global_param.data + 1.1*(user_updated.samples / total_train)*(user_new_param.data - user_old_param.data)
                    else:
                        global_param.data = global_param.data + 0.99*(user_updated.samples / total_train)*(user_new_param.data - user_old_param.data)
                    user_old_param.data = user_new_param.data.clone()
                self.test()
                # for global_param, user_old_param, user_new_param in zip(self.model.parameters(), self.users[user_updated.id].model, user_updated.model):
                #     global_param.data = global_param.data - (user_updated.samples / total_train)*(user_old_param.data - user_new_param.data)
                #     user_old_param.data = user_new_param.data.clone()
        else:
            for user_updated in user_data:
                self.users[user_updated.id].samples = user_updated.samples
                for new_param, old_param in zip(user_updated.model, self.users[user_updated.id].model):
                    old_param.data = new_param.data.clone()
            total_train = 0
            for user in self.users.values():
                total_train += user.samples
            for index, global_copy in enumerate(self.model_cpoy):
                global_copy.data = torch.zeros_like(global_copy.data)
                for user in self.users.values():
                    global_copy.data = global_copy.data + (user.samples / total_train)*user.model[index].data
            for global_param, global_copy in zip(self.model.parameters(), self.model_cpoy):
                global_param.data = global_copy.data.clone()