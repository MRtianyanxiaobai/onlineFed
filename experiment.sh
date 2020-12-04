#/bin/bash

gpu_check=$(nvidia-smi -q -d MEMORY|grep -E 'Free'|head -1|awk '{print $3}')
echo $gpu_check
while [ "$gpu_check" -le 9000 ]; do
    sleep 30m
done
echo 'Start FedAvg'
source start.sh FedAvg

while [ "$gpu_check" -le 9000 ]; do
    sleep 30m
done
echo 'Start ASO'
source start.sh ASO

while [ "$gpu_check" -le 9000 ]; do
    sleep 30m
done
echo 'Start FAFed'
source start.sh FAFed