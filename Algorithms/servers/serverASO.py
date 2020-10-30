import torch
import os

from Algorithms.servers.serverBase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

class ServerASO(Server):
    def __init__(self, algorithm, model, test_data):
        super().__init__(algorithm, model, test_data)
    
    def aggregate_parameters(self, user_updated):
        self.users[user_updated.id].samples = user_updated.samples
        total_train = 0
        for user in self.users:
            total_train += user.samples
        for global_param, user_old_param, user_new_param in zip(self.model.parameters(), self.users[user_updated.id].model.parameters(), user_updated.model):
            global_param.data = global_param.data - (sample_len / total_train)*(user_old_param.data - user_new_param.data)
            user_old_param.data = user_new_param.data.clone()


