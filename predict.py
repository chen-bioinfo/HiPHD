import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import sys
import math
import pprint
import random
import yaml
import shutil
import numpy as np
import argparse

import torch
from torch.optim import lr_scheduler

from torchdrug import core, models, tasks, datasets, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import dataset, model,tasks, engine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--esm", help="whether to enable protein language model, set 1 to enable, set 0 to disable",type=bool ,required=True)
    parser.add_argument("-s", "--system", help="protein classification system, 'scope' or 'cath' ", type=str, default='scope', required=True)
    parser.add_argument("-g", "--gpus", help="whether to use gpu and specify the index of gpu. Set -1 to use CPU", type=int, default=-1)
 

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    # vars = {}
    # parser = argparse.ArgumentParser()
    # for var in vars:
    #     parser.add_argument("--%s" % var, default="null")
    # vars = parser.parse_known_args(unparsed)[0]
    # vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args

# def train_and_validate(cfg, solver, scheduler):
#     if cfg.train.num_epoch == 0:
#         return

#     # step = math.ceil(cfg.train.num_epoch / 50)
#     # step = 1 #  FOR TEST ONLY
#     step = cfg.train.save_step
#     best_result = float("-inf")
#     best_epoch = -1

#     # 冻结参数
#     # for n, p in solver.model.named_parameters():
#     #     if 'mlp' not in n:
#     #         p.requires_grad = False

#     for i in range(0, cfg.train.num_epoch, step):
#         kwargs = cfg.train.copy()
#         kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
#         del kwargs["save_step"]
        
#         solver.train(**kwargs)
#         solver.save("model_epoch_%d.pth" % solver.epoch)
#         metric = solver.evaluate("valid")
#         # solver.evaluate("valid")

#         result = metric[cfg.metric]
#         if result > best_result:
#             best_result = result
#             best_epoch = solver.epoch
#         if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(result)

#     solver.load("model_epoch_%d.pth" % best_epoch)
#     return solver
def init_model():
    model_base = None

def test(cfg, solver):
    return solver.evaluate("test")


if __name__ == "__main__":
    args = parse_args()
    # working_dir = util.create_working_directory(cfg)
    model_path = ''
    if args['system'] == 'scope' and args['esm']:
        model_path = 'final_model/hiphd_esm_scope'
    elif args['system'] == 'scope' and ~args['esm']:
        model_path = 'final_model/hiphd_scope'
    elif args['system'] == 'cath' and args['esm']:
        model_path = 'final_model/hiphd_esm_cath'
    else:
        model_path = 'final_model/hiphd_cath'

    task = core.Configurable.load_config_dict(os.path.join(model_path, 'config.yaml'))
    state = torch.load(os.path.join(model_path, 'model.pth'), map_location=args.device)
    task.load_state_dict(state['model'])

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, scheduler = util.build_downstream_solver(cfg, dataset)

    train_and_validate(cfg, solver, scheduler)
    test(cfg, solver)
