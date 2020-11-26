import torch
import os
import copy
from Algorithms.servers.serverBase import Server
import numpy as np

class ServerFedAvg(Server):
    def __init__(self, algorithm, model, async_process, test_data):
        super().__init__(algorithm, model[0], async_process, test_data)
    
    def aggregate_parameters(self, user_data):
        if self.async_process == True:
            for user_updated in user_data:  
                self.users[user_updated.id].samples = user_updated.samples
                total_train = 0
                for user in self.users.values():
                    total_train += user.samples
                print("total train", total_train)
                # for global_param, user_old_param, user_new_param in zip(self.model.parameters(), self.users[user_updated.id].model, user_updated.model):
                #     global_param.data = global_param.data - (user_updated.samples / total_train)*(user_old_param.data - user_new_param.data)
                print(user_updated.id, 'aggerated')
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
            
        