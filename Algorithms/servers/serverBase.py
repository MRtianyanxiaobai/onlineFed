import torch
import os
import numpy as np
import copy
from torch.utils.data import DataLoader

from utils.model_utils import Object
can_gpu = torch.cuda.is_available()

class Server:
    def __init__(self, algorithm, model, async_process, test_data):
        self.model = copy.deepcopy(model)
        self.algorithm = algorithm
        self.test_data = test_data
        self.async_process = async_process
        # self.testloader = DataLoader(test_data, len(test_data))
        self.test_acc = 0

        self.status = False
        self.update_list = []
        self.users = {}

    def append_user(self, user):
        self.users[user.id] = Object(dict(id=user.id, model=copy.deepcopy(list(user.model.parameters())),samples=user.train_data_samples))

    def update_parameters(self, id, new_parameters, sample_len):
        print("Get update ", id)
        self.append_update_cache(id, new_parameters, sample_len)
        if self.async_process == True:
            self.clear_update_cache()
    
    def append_update_cache(self, id, new_parameters, samples_len):
        self.update_list.append(Object(dict(id=id, model=new_parameters, samples=samples_len)))

    def clear_update_cache(self):
        cache = self.update_list[:]
        self.update_list = []
        self.status = True
        self.aggregate_parameters(cache)
        # for user_data in cache:
        #     self.aggregate_parameters(user_data)
        self.status = False
        if len(self.update_list) != 0:
            print("clear cache again")
            self.clear_update_cache()
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.test_acc = test_acc*1.0 / y.shape[0]
        return test_acc, y.shape[0]
    
    def save_model(self):
        model_path = os.path.join("models")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", "server" + ".pt"))