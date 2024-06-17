import os
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
from torchdrug import core
import numpy as np
from hiphd.dataset import Scope, ScopeHB, CATH

import logging
logging.disable()

import torch
from torchdrug import core

import util

from torchdrug import data, utils, transforms
from torchdrug.core import Registry as R
from torch.utils import data as torch_data

# Assign the path of dataset and model weights
base_path = os.path.dirname(__file__)
MODEL_FILE = os.path.join(base_path,'log_ckpt/SuperfamilyPrediction/SCOPe/GearNetIEConv/2023-10-24-17-57-17/model_epoch_100.pth')
SCOPE_FILE = os.path.join(base_path, 'DataTest')

def evaluate(test_set,model):
    model.eval()
    preds = []
    targets = []
    dataloader = data.DataLoader(test_set, batch_size= 1)

    for batch in dataloader:
        batch = utils.cuda(batch)
        pred, target = model.predict_and_target(batch)

        preds.append(pred)
        targets.append(target)

    pred = utils.cat(preds)
    target = utils.cat(targets)
    metric = model.evaluate(pred, target)
    return metric

if __name__ == '__main__':
    print('start evaluation')

    cfg = util.load_config('config/balance_scope.yaml')

    cfg.task.task = ['fold_label']
    task = core.Configurable.load_config_dict(cfg.task)


    state = torch.load(MODEL_FILE, map_location='cuda')
    task.load_state_dict(state["model"], strict=False)
    task = task.to('cuda')

    origin_data = Scope(SCOPE_FILE, transform = transforms.ProteinView('residue'))
    test = origin_data.split()[2]
    valid = origin_data.split()[1]
    train = origin_data.split()[0]

    valid_acc = evaluate(valid, task)
    test_acc = evaluate(test, task)
