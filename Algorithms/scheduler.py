import torch
import os
import json
import numpy as np
import copy
from Algorithms.servers.serverASO import ServerASO
from Algorithms.servers.serverFedAvg import ServerFedAvg
from Algorithms.servers.serverFAFed import ServerFAFed
from Algorithms.users.userASO import UserASO
from Algorithms.users.userFedAvg import UserFedAvg
from Algorithms.users.userFAFed import UserFAFed
from utils.model_utils import read_data, read_user_data
import torch
import pandas as pd
torch.manual_seed(0)


class Scheduler:
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate, lamda, beta, num_glob_iters,
                 local_epochs, optimizer, num_users, times, data_load):
        self.dataset = dataset
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.beta = beta
        self.times = times
        self.data_load = data_load
        self.num_users = num_users
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.users = []
        self.avg_local_acc = []
        self.avg_local_train_acc = []
        self.avg_local_train_loss = []
        self.server_acc = []

        data = read_data(dataset)
        total_users = len(data[0])
        self.num_users = min(total_users, num_users)
        test_data = []
        for i in range(self.num_users):
            id, train, test = read_user_data(i, data, dataset)
            if algorithm == 'FedAvg':
                user = UserFedAvg(id, train, test, model, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            if algorithm == 'ASO':
                user = UserASO(id, train, test, model, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            if algorithm == 'FAFed':
                user = UserFAFed(id, train, test, model, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            self.users.append(user)
            test_data.extend(test)
        if algorithm == 'FedAvg':
            self.server = ServerFedAvg(algorithm, model, test_data)
        if algorithm == 'ASO':
            self.server = ServerASO(algorithm, model, test_data)
        if algorithm == 'FAFed':
            self.server = ServerFAFed(algorithm, model, test_data)
        for user in self.users:
            self.server.append_user(user)
    
    def run(self):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            new_data_flag = torch.rand(self.num_users)
            activation_users = []
            new_data_num = []
            for index, val in enumerate(new_data_flag):
                if val < 0.5:
                    activation_users.append(self.users[index])
                    new_data_num.append(int(val*10))
            for user, new_data in zip(activation_users, new_data_num):
                user.train(new_data, self.server)
            self.evaluate()
        self.save_results()
        self.server.save_model()
    
    def evaluate(self):
        self.evaluate_users()
        self.evaluate_server()

    def evaluate_users(self):
        stats = self.users_test()  
        stats_train = self.users_train_error_and_loss()
        avg_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.avg_local_acc.append(avg_acc)
        self.avg_local_train_acc.append(train_acc)
        self.avg_local_train_loss.append(train_loss)
        print("Average Local Accurancy: ", avg_acc)
        print("Average Local Trainning Accurancy: ", train_acc)
        print("Average Local Trainning Loss: ",train_loss)

    def evaluate_server(self):
        stats = self.server.test()
        server_acc = stats[0]*1.0/stats[1]
        self.server_acc.append(server_acc)
        print("Central Model Accurancy: ", server_acc)
    
    def users_test(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def users_train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses
    
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs) + "_" + self.data_load
        alg = alg + "_" + str(self.times)
        if (len(self.avg_local_acc) != 0 &  len(self.avg_local_train_acc) & len(self.avg_local_train_loss)) :
            dictData={}
            dictData['avg_local_acc'] = self.avg_local_acc[:]
            dictData['avg_local_train_acc'] = self.avg_local_train_acc[:]
            dictData['avg_local_train_loss'] = self.avg_local_train_loss[:]
            dictData['central_model_acc'] = self.server_acc[:]
            dataframe = pd.DataFrame(dictData)
            fileName = "./results/"+alg+'.csv'
            dataframe.to_csv(fileName, index=False, sep=',')
    

    
        




         
