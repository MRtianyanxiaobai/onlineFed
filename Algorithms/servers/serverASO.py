import torch
import os

from Algorithms.servers.serverBase import Server
import numpy as np

class ServerASO(Server):
    def __init__(self, algorithm, model, async_process, test_data):
        super().__init__(algorithm, model[0], async_process, test_data)
    
    def aggregate_parameters(self, user_data):
        if self.async_process == True:
            for user_updated in user_data:  
                self.users[user_updated.id].samples = user_updated.samples
                total_train = 0
                for user in self.users.values():
                    total_train += user.samples
                #模型聚合
                for global_param, user_old_param, user_new_param in zip(self.model.parameters(), self.users[user_updated.id].model, user_updated.model):
                    global_param.data = global_param.data - (user_updated.samples / total_train)*(user_old_param.data - user_new_param.data)
                    user_old_param.data = user_new_param.data.clone()
                # # feature learning
                # alpha = torch.exp(torch.abs(list(self.model.parameters())[0].data))
                # for index, val in enumerate(alpha):
                #     sumCol = torch.sum(val)
                #     alpha[index] = torch.div(val, sumCol.item())
                # for index, global_param in enumerate(self.model.parameters()):
                #     if index == 0:
                #         global_param.data = global_param.data.mul(alpha)
        else:
            for user_updated in user_data:
                self.users[user_updated.id] = user_updated
            total_train = 0
            for user in self.users.values():
                total_train += user.samples
            # 模型聚合
            for user in self.users.values():
                for global_param, local_param in zip(self.model.parameters(), user.model):
                    global_param.data = (user.samples / total_train)*local_param.data

        


