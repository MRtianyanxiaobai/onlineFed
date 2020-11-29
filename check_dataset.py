import argparse
from utils.model_utils import read_data_async
def main(dataset, num_users, user_labels, niid):
    read_data_async(dataset, niid, num_users, user_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FasionMNIST", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["mclr", "cnn"])
    parser.add_argument("--async_process", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--lamda", type=float, default=1.0, help="Regularization term")
    parser.add_argument("--beta", type=float, default=0.001, help="Decay Coefficient")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["FedAvg", "ASO", "FAFed"]) 
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--user_labels", type=int, default=5, help="Number of Labels per client")
    parser.add_argument("--niid", type=str2bool, default=True, help="data distrabution for iid or niid")
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
        numusers = args.numusers,
        user_labels = args.user_labels,
        niid = args.niid,
        )