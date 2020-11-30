import zerorpc
import os
import argparse
import torch
import gevent
from Algorithms.servers.serverASO import ServerASO
from Algorithms.servers.serverFedAvg import ServerFedAvg
from Algorithms.servers.serverFAFed import ServerFAFed
from utils.model_utils import read_test_data_async
from Algorithms.models.model import *
import pandas as pd
torch.manual_seed(0)
class RpcController(object):
    def __init__(self, dataset, algorithm, model, num_users, async_process, batch_size):
        self.dataset = dataset
        test_data = read_test_data_async(dataset)
        if(model == "mclr"):
            if(dataset == "MNIST"):
                pre_model = Mclr_Logistic()
            else:
                pre_model = Mclr_Logistic(60,10)
        if(model == "cnn"):
            if(dataset == "MNIST"):
                pre_model = Net()
            else:
                pre_model = CifarNet()
        pre_model = pre_model.cuda()
        pre_model.eval()
        if algorithm == 'FedAvg':
            server = ServerFedAvg(algorithm, pre_model, async_process, test_data, batch_size)
        if algorithm == 'ASO':
            server = ServerASO(algorithm, pre_model, async_process, test_data, batch_size)
        if algorithm == 'FAFed':
            server = ServerFAFed(algorithm, pre_model, async_process, test_data, batch_size)
        # server.test()
        self.server = server
        self.nun_users = num_users
        self.server.save_model()
        self.client_list = {}
        self.client_counter = 0
        self.aggerate_counter = 0
        # self.read = {}
        # self.write = False
        # self.start_train = False
    def client_update(self, id, samples_len):
        try:
            # if self.start_train is False:
            #     print('not start')
            #     return False
            userModel = self.server.load_model("client_"+id)
            self.server.update_parameters(id, userModel, samples_len)

            self.server.save_model()
            self.aggerate_counter = self.aggerate_counter + 1
            if self.aggerate_counter % 10 == 0:
                self.server.test()
                print('Server updated ', id, ', test_acc is ', self.server.test_acc)
            else:
                print('Server updated ', id)
            return True
        except Exception as e:
            print('client update error', e)
            return False

    def add_client(self, id, samples):
        try:
            # self.read[id] = False
            self.server.save_model("server_"+id)
            self.server.append_user(id, samples)
            self.client_counter = self.client_counter + 1
            # if self.client_counter == self.nun_users:
            #     self.start_train = True
            return True
        except Exception as e:
            print('add client error', e)
            return False

    def get_model(self, id):
        for global_param, user_init in zip(self.server.model.parameters(), self.server.users[id].model):
            user_init.data = global_param.data.clone()
        self.server.save_model("server_"+id)
        return True

    def close_client(self, id):
        self.client_counter = self.client_counter - 1
        if self.client_counter == 0:
            dictData = {}
            dictData['server_test_acc'] = self.server.test_acc_log[:]
            dataFrame = pd.DataFrame(dictData)
            filename = './results/'+self.server.algorithm+'_'+self.dataset+'_'+'.csv'
            dataFrame.to_csv(filename, index=False, sep=',')
            print('Server Over !')

    
    
        
def main(dataset, algorithm, model, num_users, async_process, batch_size):
         
    server = zerorpc.Server(RpcController(dataset, algorithm, model, num_users, async_process, batch_size))
    server.bind('tcp://0.0.0.0:8888')
    # publisher = zerorpc.Publisher()
    # publisher.bind('tcp://0.0.0.0:8889')
    print('Server Start!')
    # gevent.spawm(server.run)
    server.run()

         
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):

        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FasionMNIST", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["mclr", "cnn"])
    parser.add_argument("--async_process", type=str2bool, default=True)
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["FedAvg", "ASO", "FAFed"]) 
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    main(dataset=args.dataset, algorithm=args.algorithm, model=args.model, num_users=args.numusers, async_process=args.async_process, batch_size=args.batch_size)

