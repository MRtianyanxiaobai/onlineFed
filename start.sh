#!/bin/sh
if [ $1 ]; then
    Algorithm=$1
else
    Algorithm='FedAvg'
fi

Dataset='MNIST'         # MNIST Cifar10 FashionMNIST
Model='cnn'             # cnn mclr
Async_process='True'    # True False
# Algorithm=      # FedAvg ASO FAFed
Optimizer='SGD'         # SGD
Batch_size=20    
Lr=0.008
Lamda=0.5
Beta=0.1
Num_global_iters=800
Local_epochs=20
Num_users=5
User_labels=5
Niid='True'            # True False
Data_load='fixed'      # fixed flow

# output='python3 check_environment.py'
gnome-terminal -- tmux new -t server \; send-keys 'python3 server.py --dataset='$Dataset' --model='$Model' --async_process='$Async_process' --algorithm='$Algorithm' --batch_size='$Batch_size C-m \;

index=0
while [ "$index" -lt "$Num_users" ]; do
    gnome-terminal -- tmux new -t client$index \; send-keys 'python3 client.py --dataset='$Dataset' --model='$Model' --async_process='$Async_process' --batch_size='$Batch_size' --learning_rate='$Lr' --lamda='$Lamda' --beta='$Beta' --num_global_iters='$Num_global_iters' --optimizer='$Optimizer' --local_epochs='$Local_epochs' --algorithm='$Algorithm' --data_load='$Data_load' --index='$index C-m \;
    index=$((index + 1))
done