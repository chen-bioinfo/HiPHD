# NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_nod=2  --master_port=29501 downstream.py  -c config/hbond_mt_plm.yaml --gpus [0,1] --ckpt ~/Superfamily/GearNet/log_ckpt/MultiTaskPrediction/ScopeHBMultiTask/FusionNetwork/2023-12-23-12-25-18/model_epoch_10.pth

# NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_nod=2  --master_port=29500 downstream.py  -c config/cath/hbond_mt_plm.yaml --gpus [0,1]
NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_nod=2  --master_port=29500 downstream.py  -c config/abl/cath/hb_spatial_seq.yaml --gpus [0,1] --ckpt /geniusland/home/qufuchuan/Superfamily/GearNet/seed_ckpt/MultiTaskPrediction/HBCATH/GearNetIEConv/2024-01-30-16-50-16/model_epoch_30.pth 

# NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_nod=2  --master_port=29500 downstream.py  -c config/cath/gear_plm.yaml --gpus [2,3]

# python downstream.py -c config/abl/cath/hb_spatial_seq.yaml --gpus [6] --ckpt /geniusland/home/qufuchuan/Superfamily/GearNet/seed_ckpt/MultiTaskPrediction/HBCATH/GearNetIEConv/2024-01-30-16-50-16/model_epoch_30.pth 
# #python downstream.py -c config/balance_scope.yaml --gpus [1]
