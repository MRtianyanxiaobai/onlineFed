import torch
import os
import copy
from Algorithms.servers.serverBase import Server
import numpy as np

class ServerFAFed(Server):
    def __init__(self, algorithm, model, test_data):
        super().__init__(algorithm, model[0], test_data)
    
    def aggregate_parameters(self, user_updated):
        self.users[user_updated.id].samples = user_updated.samples
        total_train = 0
        for user in self.users.values():
            total_train += user.samples
        self.model_copy = copy.deepcopy(list(self.model.parameters()))
        for global_param, user_old_param, user_new_param in zip(self.model.parameters(), self.users[user_updated.id].model, user_updated.model):
            global_param.data = global_param.data - (user_updated.samples / total_train)*(user_old_param.data - user_new_param.data)
            user_old_param.data = user_new_param.data.clone()
        updated_stats = self.test()
        updated_acc = updated_stats[0]*1.0/updated_stats[1]
        if updated_acc < self.test_acc:
            for updated_param, old_param in zip(self.model.parameters(), self.model_copy, self.users[user_updated.id].model, user_updated.model):
                updated_param.data = old_param.data - 0.5*(user_updated.samples / total_train)*(user_old_param.data - user_new_param.data)
        