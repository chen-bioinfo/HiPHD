
# NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_nod=2  --master_port=29500 downstream.py  -c config/cath/hbond_mt_plm.yaml --gpus [0,1]
NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_nod=2  --master_port=29500 downstream.py  -c config/cath/hbond_plm.yaml --gpus [0,1] 

#python downstream.py -c config/balance_scope.yaml --gpus [1]
