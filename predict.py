import os
import sys
import math
import pprint
import random
import yaml
import shutil
import json
import argparse
import numpy as np
import torch
from torch.optim import lr_scheduler
from torchdrug import core, models, tasks, datasets, utils, data, transforms
from torchdrug.utils import comm
import util
from util_predict import load_pdb
from hiphd import tasks
from gearnet import layer, model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="one single pdb file",type=str, default= None, required=True)
    parser.add_argument("-e", "--esm", action='store_true', help="whether to enable protein language model")
    parser.add_argument("-s", "--system", help="protein classification system, 'scope' or 'cath' ", type=str, default='scope')
    parser.add_argument("-g", "--gpus", help="whether to use gpu and specify the index of gpu. Set -1 to use CPU", type=int, default=-1)

    args = parser.parse_args()

    return args

def init_model(args, device):
    if args.system == 'scope' and args.esm:
        model_path = 'model/hiphd_esm_scope'
    elif args.system == 'scope' and ~args.esm:
        model_path = 'model/hiphd_scope'
    elif args.system == 'cath' and args.esm:
        model_path = 'model/hiphd_esm_cath'
    else:
        model_path = 'model/hiphd_cath'

    cfg = util.load_config(model_path+'.yaml', {})
    task = core.Configurable.load_config_dict(cfg.task)
    state = torch.load(model_path+'.pth', map_location=device)
    task.load_state_dict(state)
    return task

def init_map(args):
    if args.system == 'scope':
        with open('map/lab2idx_scop.json', 'r') as fp:
            classmap = json.load(fp)['super']
    else:
        with open('map/lab2idx_cath.json', 'r') as fp:
            classmap = json.load(fp)['h']
    classmap = {v:k for k,v in classmap.items()}
    return classmap

if __name__ == "__main__":
    args = parse_args()
    if args.gpus != -1:
        device = f'cuda:{args.gpus}'
    else:
        device = 'cpu'

    base_model = init_model(args, device)
    classmap = init_map(args)

    prot_data = load_pdb(args.file)
    input_data = data.Protein.pack([prot_data['graph']])
    input_data = {'graph' : input_data}

    base_model.eval()
    with torch.no_grad():
        res = base_model.predict(input_data)[-1].argmax()
    
    res_label = classmap[res.item()]
    print(f'The result of {args.file} is {res_label}')