import os
import sys
import logging
from itertools import islice

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.core import Registry as R
from torchdrug.utils import comm, pretty
from torchdrug.core import Engine
from torch.cuda.amp import autocast, GradScaler


module = sys.modules[__name__]
logger = logging.getLogger(__name__)

@R.register("core.EngineWithAMP")
class EngineWithAMP(Engine):
    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, batch_size=1, gradient_interval=1, num_worker=0, logger="logging", log_interval=100):
        super().__init__(task, train_set, valid_set, test_set, optimizer, scheduler, gpus, batch_size, gradient_interval, num_worker, logger, log_interval)
        
    def train(self, num_epoch=1, batch_per_epoch=None):
        scaler = GradScaler()
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                with autocast():
                    loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                scaler.scale(loss).backward()
                # loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    # self.optimizer.step()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()

@R.register("core.EngineWithFGM")
class EngineWithFGML(Engine):
    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, batch_size=1, gradient_interval=1, num_worker=0, epsilon_fgm = 0.3,logger="logging", log_interval=100):
        super().__init__(task, train_set, valid_set, test_set, optimizer, scheduler, gpus, batch_size, gradient_interval, num_worker, logger, log_interval)
        self.epsilon_fgm = epsilon_fgm

    def train(self, num_epoch=1, batch_per_epoch=None):
        scaler = GradScaler()
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)
                
                batch['graph'].node_position.requires_grad = True

                with autocast():
                    loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                scaler.scale(loss).backward()
                # loss.backward()
                metrics.append(metric)

                # Adversarial Learning
                batch['graph'].node_position = batch['graph'].node_position + self.epsilon_fgm*batch['graph'].node_position.grad.sign()
                with autocast():
                    loss, metric = model(batch)
                loss = loss / gradient_interval
                scaler.scale(loss).backward()

                if batch_id - start_id + 1 == gradient_interval:
                    # self.optimizer.step()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()
