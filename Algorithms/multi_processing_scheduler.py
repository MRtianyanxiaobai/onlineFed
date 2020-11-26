import torch
import os
import json
import time
import numpy as np
import copy
from Algorithms.servers.serverASO import ServerASO
from Algorithms.servers.serverFedAvg import ServerFedAvg
from Algorithms.servers.serverFAFed import ServerFAFed
from Algorithms.users.userASO import UserASO
from Algorithms.users.userFedAvg import UserFedAvg
from Algorithms.users.userFAFed import UserFAFed
from utils.model_utils import read_data_async, read_user_data_async, read_test_data_async
import pandas as pd
import torch.multiprocessing as mp
torch.manual_seed(0)
task_counter = 10
task_counter_flag = False
over_flag = False

def client_task(i, algorithm, dataset, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, num_glob_iters, update_queue, queue_lock):
    id, train, test = read_user_data_async(i, dataset)
    if algorithm == 'FedAvg':
        user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    if algorithm == 'ASO':
        user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    if algorithm == 'FAFed':
        user = UserFAFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    print(user.id, " Start !")
    for i in range(num_glob_iters):
        print(user.id, 'run ', i)
        if user.can_train() == False:
            continue
        else:
            if user.check_async_update():
                queue_lock.acquire()
                update_queue.put([user.id, list(user.model.parameters()), user.train_data_samples])
                queue_lock.release()
                user.trained = False
        time_offset = time.sleep(torch.randint(10, 100, (1,)).item() / 1000)
        global_model = user.get_global_parameters()
        update_queue.put([user.id])
        user.train(global_model)
        print(user.id, "over in ", i)
        if user.check_async_update():
            queue_lock.acquire()
            update_queue.put([user.id, list(user.model.parameters()), user.train_data_samples])
            queue_lock.release()
            user.trained = False

    while(task_counter_flag):
        print(user.id, " wait flag.")
    task_counter_flag = True
    task_counter = task_counter - 1
    task_counter_flag = False

def server_task(algorithm, dataset, model, async_process, update_queue):
    test_data = read_test_data_async(dataset)
    if algorithm == 'FedAvg':
        server = ServerFedAvg(algorithm, model, async_process, test_data)
    if algorithm == 'ASO':
        server = ServerASO(algorithm, model, async_process, test_data)
    if algorithm == 'FAFed':
        server = ServerFAFed(algorithm, model, async_process, test_data)
    print('Server Start !')
    over_flag = False
    while(task_counter > 0):
        while (update_queue.qsize() > 0):
            print('get a new data')
            new_data = update_queue.get()
            if len(new_data) == 1:
                for local_param, global_param in zip(server.users[new_data[0]].model, server.model.parameters()):
                    local_param.data = global_param.data.clone()
                # server.users[new_data[0]].model = copy.deepcopy(server.model.parameters())
            else:
                server.update_parameters(new_data[0], new_data[1], new_data[2])
                server.test()
                print('Server updated, test_acc is ', server.test_acc)
    print('Server Over.')
    over_flag = True
update_queue = mp.Queue()
queue_lock = mp.Lock()
class Scheduler:
    def __init__(self, dataset,algorithm, model, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters,
                 local_epochs, optimizer, num_users, user_labels, niid, times, data_load):
        self.dataset = dataset
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.async_process = async_process
        self.lamda = lamda
        self.beta = beta
        self.times = times
        self.data_load = data_load
        self.num_users = num_users
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.user_labels = user_labels
        self.niid = niid
        self.users = []
        self.users_process = []
        task_counter = num_users
        model[0].share_memory()
        # data split
        data = read_data_async(dataset, niid, num_users, user_labels)
        self.num_users = num_users
        test_data = []
        for i in range(self.num_users):
            userProcess = mp.Process(target=client_task, args=(i, algorithm, dataset, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, num_glob_iters, update_queue, queue_lock))
            self.users_process.append(userProcess)
            # id, train, test = read_user_data(i, data, dataset)
            # if algorithm == 'FedAvg':
            #     user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            # if algorithm == 'ASO':
            #     user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            # if algorithm == 'FAFed':
            #     user = UserFAFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            # self.users.append(user)
            # test_data.extend(test)
        # if algorithm == 'FedAvg':
        #     server = ServerFedAvg(algorithm, model, async_process, test_data)
        # if algorithm == 'ASO':
        #     server = ServerASO(algorithm, model, async_process, test_data)
        # if algorithm == 'FAFed':
        #     server = ServerFAFed(algorithm, model, async_process, test_data)
        # for user in self.users:
        #     server.append_user(user)
        # self.server = server
        self.server_process = mp.Process(target=server_task, args=(algorithm, dataset, model, async_process, update_queue))
            
    
    def run(self): 
        self.server_process.start()
        for user in self.users_process:
            user.start()
        
        for user in self.users_process:
            user.join()
        self.server_process.join()
        print('voier')

        
        