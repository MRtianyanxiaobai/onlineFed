import torch
import os

from Algorithms.servers.serverBase import Server
import numpy as np

class ServerFedAvg(Server):
    def __init__(self, algorithm, model, test_data):
        super().__init__(algorithm, model[0], test_data)
    
    def aggregate_parameters(self, user_updated):
        self.users[user_updated.id].samples = user_updated.samples
        total_train = 0
        for user in self.users.values():
            total_train += user.samples
        for global_param, user_old_param, user_new_param in zip(self.model.parameters(), self.users[user_updated.id].model, user_updated.model):
            global_param.data = global_param.data - (user_updated.samples / total_train)*(user_old_param.data - user_new_param.data)
            user_old_param.data = user_new_param.data.clone()