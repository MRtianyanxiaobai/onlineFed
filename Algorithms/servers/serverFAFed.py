import torch
import os
import copy
from Algorithms.servers.serverBase import Server

class ServerFAFed(Server):
    def __init__(self, algorithm, model, async_process, test_data, batch_size):
        super().__init__(algorithm, model, async_process, test_data, batch_size)
        self.last_model = copy.deepcopy(list(model.parameters()))
        self.new_model = copy.deepcopy(list(model.parameters()))
        self.benefit = True
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for i, (x, y) in enumerate(self.testloader):
            output = self.model(x.cuda())
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y.cuda())).item()
        last_acc = self.test_acc
        self.test_acc = test_acc*1.0 / len(self.test_data)
        self.test_acc_log.append(self.test_acc)
        if self.test_acc - last_acc >= 0:
            self.benefit = True
        else:
            self.benefit = False
        return test_acc, len(self.test_data)
    
    def aggregate_parameters(self, user_data):
        if self.async_process == True:
            for user_updated in user_data:
                self.users[user_updated.id].samples = user_updated.samples
                total_train = 0
                for user in self.users.values():
                    total_train += user.samples
                for global_param, last_global_param, user_new_param, user_old_param in zip(self.model.parameters(), self.last_model, user_updated.model, self.users[user_updated.id].model):
                    distance = user_new_param.data - user_old_param.data
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
                        global_param.data = global_param.data + 0.9*(user_updated.samples / total_train)*(user_new_param.data - user_old_param.data)
        else:
            for user_updated in user_data:
                self.users[user_updated.id].samples = user_updated.samples
                for new_param, old_param in zip(user_updated.model, self.users[user_updated.id].model):
                    old_param.data = new_param.data.clone()
            total_train = 0
            for user in self.users.values():
                total_train += user.samples
            for index, global_copy in enumerate(self.new_model):
                global_copy.data = torch.zeros_like(global_copy.data)
                for user in self.users.values():
                    global_copy.data = global_copy.data + (user.samples / total_train)*user.model[index].data
            for new_param, current_param, last_param in zip(self.new_model, self.model.parameters(), self.last_model):
                distance = new_param.data - current_param.data
                last_distance = current_param.data - last_param.data
                distance_vec = torch.flatten(distance)
                last_distance_vec = torch.flatten(last_distance)
                similarity = torch.cosine_similarity(last_distance_vec, distance_vec, dim=0).item()
                last_param.data = current_param.data.clone()
                if self.benefit is True and similarity >= 0:
                    current_param.data = current_param.data + 1.1*distance
                elif self.benefit is False and similarity <= 0:
                    current_param.data = current_param.data + 1.1*distance
                else:
                    current_param.data = current_param.data + 0.9*distance
