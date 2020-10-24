#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from Algorithms.servers.fedAvg import FedAvg
from Algorithms.models.model import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, times, user_data_type):

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic(), model
            else:
                model = Mclr_Logistic(60,10), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net(), model
            elif(dataset == "Cifar10"):
                model = CifarNet(), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN(), model
            else: 
                model = DNN(60,20,10), model

        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(dataset, algorithm, model, batch_size, learning_rate, lamda, num_glob_iters, local_epochs, optimizer, numusers, i)

        server.train()
        server.test()

    # save data
    if (algorithm == 'FedAvg'):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K,times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["FedAvg", "FedCache"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--user_data_type", type=str, default='fixed', choices=["fixed", "cache", 'flow'], help="user data load type")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        times = args.times
        user_data_type=args.user_data_type
        )