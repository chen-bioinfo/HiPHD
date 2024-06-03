import os
import sys
import subprocess

seeds = [22, 23, 24, 25, 26]
configs = os.listdir('config/')

for config in configs:
    for seed in seeds:
        os.system(f"NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_nod=2  --master_port=29501 downstream.py  -c config/{config} -s {seed} --gpus [0,1]")