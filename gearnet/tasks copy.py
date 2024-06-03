import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from torch.cuda import amp


@R.register("tasks.SuperfamilyPrediction")
class SuperfamilyPrediction(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="ce", metric=("mae", "rmse"), num_mlp_layer=1,
                mlp_batch_norm = True,mlp_dropout = 0,
                 num_class=None, graph_construction_model=None, verbose=0):
        super(SuperfamilyPrediction, self).__init__()

        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        # For classification tasks, we disable normalization tricks.
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        # hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        # self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)],
        #                     batch_norm = self.mlp_batch_norm, dropout = self.mlp_dropout)

    def preprocess(self, train_set, valid_set, test_set):
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class
        self.weight = torch.as_tensor(weight, dtype=torch.float)

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)],
                            batch_norm = self.mlp_batch_norm, dropout = self.mlp_dropout)

    @amp.autocast()
    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        for criterion, weight in self.criterion.items():
            loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            loss = functional.masked_mean(loss, labeled, dim=0)
            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])

        return pred

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for s ,t in zip(score, self.task):
                metric["%s [%s]" % (name, t)] = s

        return metric

