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

def client_task(i, algorithm, dataset, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, num_glob_iters, update_queue, download_queue, update_lock, download_lock, task_counter):
    id, train, test = read_user_data_async(i, dataset)
    if algorithm == 'FedAvg':
        user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    if algorithm == 'ASO':
        user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    if algorithm == 'FAFed':
        user = UserFAFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    print(user.id, " Start !")
    update_lock.acquire()
    update_queue.put([user.id, list(user.model.parameters()), user.train_data_samples])
    update_lock.release()
    for i in range(num_glob_iters):
        if user.can_train() == False:
            continue
        else:
            if user.check_async_update():
                # update_lock.acquire()
                update_queue.put([user.id, list(user.model.parameters()), user.train_data_samples])
                # update_lock.release()
                user.trained = False
        time_offset = time.sleep(torch.randint(10, 100, (1,)).item() / 1000)
        # update_lock.acquire()
        update_queue.put([user.id])
        # update_lock.release()
        print(user.id, 'in', i)
        # download_lock.acquire()
        global_model = download_queue.get()
        # download_lock.release()
        print(user.id, 'get a global')
        user.train(global_model)
        print(user.id, 'trained')
        if user.check_async_update():
            # update_lock.acquire()
            update_queue.put([user.id, list(user.model.parameters()), user.train_data_samples])
            # update_lock.release()
            user.trained = False
        user.test()
    task_counter.value = task_counter.value - 1
    dictData = {}
    dictData[user.id+'_test_acc'] = user.test_acc_log[:]
    dataFrame = pd.DataFrame(dictData)
    filename = './results/'+algorithm+'_'+dataset+user.id+'_'+'.csv'
    dataFrame.to_csv(filename, index=False, sep=',')

def server_task(algorithm, dataset, model, async_process, update_queue, download_queue, task_counter):
    test_data = read_test_data_async(dataset)
    if algorithm == 'FedAvg':
        server = ServerFedAvg(algorithm, model, async_process, test_data)
    if algorithm == 'ASO':
        server = ServerASO(algorithm, model, async_process, test_data)
    if algorithm == 'FAFed':
        server = ServerFAFed(algorithm, model, async_process, test_data)
    print('Server Start !')
    while(task_counter.value > 0):
        while (update_queue.qsize() > 0):
            new_data = update_queue.get()
            if len(new_data) == 1:
                download_queue.put(list(server.model.parameters()))
                for local_param, global_param in zip(server.users[new_data[0]].model, server.model.parameters()):
                    local_param.data = global_param.data.clone()
            else:
                server.update_parameters(new_data[0], new_data[1], new_data[2])
                server.test()
                print('Server updated ', new_data[0], ', test_acc is ', server.test_acc)
    print('Server Over.')
    dictData = {}
    dictData['server_test_acc'] = server.test_acc_log[:]
    dataFrame = pd.DataFrame(dictData)
    filename = './results/'+algorithm+'_'+dataset+'_'+'.csv'
    dataFrame.to_csv(filename, index=False, sep=',')

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
        model.share_memory()
        # data split
        read_data_async(dataset, niid, num_users, user_labels)
        manager = mp.Manager()
        self.update_queue = manager.Queue()
        self.update_lock = manager.Lock()  
        self.download_queue = manager.Queue()
        self.download_lock = manager.Lock()
        self.task_counter = manager.Value('task_counter', num_users)
        self.num_users = num_users
        test_data = []
        for i in range(self.num_users):
            userProcess = mp.Process(target=client_task, args=(i, algorithm, dataset, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, num_glob_iters, self.update_queue, self.download_queue, self.update_lock, self.download_lock, self.task_counter))
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
        self.server_process = mp.Process(target=server_task, args=(algorithm, dataset, model, async_process, self.update_queue, self.download_queue, self.task_counter))
            
    
    def run(self): 
        self.server_process.start()
        time.sleep(0.5)
        for user in self.users_process:
            user.start()
        
        for user in self.users_process:
            user.join()
        self.server_process.join()
        del self.update_lock
        del self.update_queue
        print('over')

        
        