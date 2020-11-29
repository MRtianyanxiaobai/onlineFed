#/bin/bash

# gpu_check=$(nvidia-smi -q -d MEMORY|grep -E 'Free'|head -1|awk '{print $3}')

echo $1