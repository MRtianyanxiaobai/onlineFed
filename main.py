#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from Algorithms.scheduler import Scheduler
from Algorithms.models.model import *
import torch
torch.manual_seed(0)
torch.cuda.set_device(0)

def main(dataset, algorithm, model, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters,
         local_epochs, optimizer, numusers, user_labels, niid, times, data_load):

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().cuda(), model
            else:
                model = Mclr_Logistic(60,10).cuda(), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().cuda(), model
            elif(dataset == "Cifar10"):
                model = CifarNet().cuda(), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().cuda(), model
            else: 
                model = DNN(60,20,10).cuda(), model

        scheduler = Scheduler(dataset, algorithm, model, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters, local_epochs, optimizer, numusers, user_labels, niid, i, data_load)
        scheduler.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--async_process", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--lamda", type=float, default=1.0, help="Regularization term")
    parser.add_argument("--beta", type=float, default=0.001, help="Decay Coefficient")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["FedAvg", "ASO", "FAFed"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--user_labels", type=int, default=5, help="Number of Labels per client")
    parser.add_argument("--niid", type=bool, default=True, help="Data Distribution")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--data_load", type=str, default="fixed", choices=["fixed", "flow"], help="user data load")
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
        async_process=args.async_process,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lamda = args.lamda,
        beta = args.beta,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        user_labels=args.user_labels,
        niid = args.niid,
        times = args.times,
        data_load = args.data_load
        )