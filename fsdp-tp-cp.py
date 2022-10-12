from contextlib import contextmanager
import copy
import dataclasses
from functools import partial
import shutil
import os
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.distributed as dist

from torch.distributed._shard import shard_parameter
from torch.distributed._shard.api import shard_module
from torch.distributed._shard.checkpoint.planner import LoadPlan
from torch.distributed._shard.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed._shard.checkpoint.metadata import STATE_DICT_TYPE, BytesStorageMetadata, MetadataIndex, TensorStorageMetadata
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_plan.api import ShardingPlan
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec

from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed._shard.api import _shard_tensor


import torch.distributed._shard.checkpoint as dist_cp

from distcp_playground.utils import (
    traverse_state_dict,
    print_visitor,
)

from distcp_playground.dist_2d import (
    NestedTensorSaver,
    NestedTensorLoader,
    NestedRenamingTensorSaver,
    load_2d_optimizer_state_dict,
)
from torch.distributed._shard.checkpoint.resharding import(
    create_read_items,
    _create_sharded_read_items
)

from distcp_playground.run import dist_run
from distcp_playground.nested import unflatten_state_dict

"""
This example shows how to load / save models wrapped with FSDP using
dist.checkpoint and SHARDED_STATE_DICT.

This example requires this branch: https://github.com/kumpera/pytorch/tree/fsdp_tp_sharded
It has the following features merged into:
    dist.cp extensibility
    FSDP+TP integration
    sharded_optim_state_dict
    bug fixes all over the place
"""

# Tensor-Parallel degree
TP_DEGREE = 2
CHECKPOINT_DIR = f"/scratch/{os.environ['LOGNAME']}/checkpoint"

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = torch.nn.Linear(4, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.net1(x))

def p0(line):
    if dist.get_rank() == 0:
        print(line)


def module_sharding_plan(colwise_spec):
    return ShardingPlan(
        plan={
            "net1.weight": colwise_spec,
        },
        return_local_tensor=["net1"],
    )

def _params_fsdp_flat_order(m, params_sharded, tp_world_size):
    params = {}
    sharding_info = {}
    for name, param in m.named_parameters():
        if name not in params_sharded:
            params[name] = param.view(-1).size(0)
        else:
            params[name] = param.view(-1).size(0) // tp_world_size
            sharding_info[name] = (param.size(), 0 if "net1" in name else 1)
    return params, sharding_info

OPS_NOT_SHARD = []
SHARD_PARAMS = [ "net1.weight"]

def init_model():
    tp_pg, dp_pg = create_2d_process_groups(TP_DEGREE)
    model_tp = MyModel().cuda()

    sharding_spec = create_colwise_spec(tp_pg)
    sharding_plan = module_sharding_plan(sharding_spec)
    shard_module(model_tp, sharding_plan, process_group=tp_pg)
    model_tp = FSDP(model_tp, process_group=dp_pg)

    return model_tp, tp_pg, dp_pg


def create_2d_process_groups(tp_degree) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Create the process groups required by 2d parallelism.

    It creates ``tp_degree`` Data Parallel groups and 
     ``dist.get_world_size() // tp_degree`` Tensor Parallel groups.

    For example:
        Given world size 4 and tp_degree 2. It will create the following groups:
        TP: [0, 1] [2, 3]
        DP: [0, 2] [1, 3]

    Returns a tuple with the (TP, DP) ProcessGroups that the current rank belongs to.

    """
    tp_ids = []
    dp_ids = []
    for i in range(dist.get_world_size()):
        idx = i // tp_degree
        if len(tp_ids) <= idx:
            tp_ids.append([])
        tp_ids[idx].append(i)
        idx = i % tp_degree
        if len(dp_ids) <= idx:
            dp_ids.append([])
        dp_ids[idx].append(i)

    tp_pgs = [dist.new_group(ids) for ids in tp_ids]
    data_parallel_pgs = [dist.new_group(ids) for ids in dp_ids]
    tp_pg = tp_pgs[dist.get_rank() // tp_degree]
    fsdp_pg = data_parallel_pgs[dist.get_rank() % tp_degree]
    return tp_pg, fsdp_pg

def create_colwise_spec(pg):
    placements = [
        f"rank:{idx}/cuda:{dist.distributed_c10d.get_global_rank(pg, idx) % torch.cuda.device_count()}"
        for idx in range(pg.size())
    ]
    return ChunkShardingSpec(
        dim=0,
        placements=placements,
    )

def save_2d_model():
    torch.manual_seed(101)
    model_tp, tp_pg, dp_pg = init_model()

    with FSDP.summon_full_params(model_tp):
        print(f"{dist.get_rank()} :: before-save: {model_tp.net1.weight.local_tensor()}")

    dist.barrier()

    with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model_tp.state_dict()
        dist_cp.save_state_dict(
            state_dict=checkpoint,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=NestedTensorSaver())

def load_2d_model():
    torch.manual_seed(103)
    model_tp, tp_pg, dp_pg = init_model()

    dist.barrier()

    with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model_tp.state_dict()
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=dist_cp.FileSystemReader(path=CHECKPOINT_DIR),
            planner=NestedTensorLoader())

        model_tp.load_state_dict(checkpoint)

    with FSDP.summon_full_params(model_tp):
        print(f"{dist.get_rank()} :: after-load: {model_tp.net1.weight.local_tensor()}")

def save_2d_optim():
    torch.manual_seed(107)
    model_tp, tp_pg, dp_pg = init_model()
    optim_input = list(model_tp.parameters())
    optim = torch.optim.Adam(optim_input, lr=0.0001)

    model_tp(torch.rand(4).cuda()).sum().backward()

    optim.step()
    exp_avg = optim.state_dict()["state"][0]["exp_avg"]
    print(f"[[{dist.get_rank()}]] before-save state: {exp_avg}")

    optim_state = FSDP.sharded_optim_state_dict(model_tp, optim, optim_input)

    md = dist_cp.save_state_dict(
        state_dict=optim_state,
        storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        planner=NestedRenamingTensorSaver()
    )

def load_2d_optim():
    torch.manual_seed(103)
    model_tp, tp_pg, dp_pg = init_model()
    optim_input = list(model_tp.parameters())
    optim = torch.optim.Adam(optim_input, lr=0.0001)

    with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model_tp.state_dict()
        optim_state = load_2d_optimizer_state_dict(
            model_state_dict=model_state_dict,
            optimizer_prefixes={"state", "param_groups"},
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR), 
            dp_pg=dp_pg
        )

        flattened_osd = FSDP.flatten_sharded_optim_state_dict(
            optim_state, model_tp, optim_input, dp_pg
        )

        optim.load_state_dict(flattened_osd)

    exp_avg = optim.state_dict()["state"][0]["exp_avg"]
    print(f"[[{dist.get_rank()}]] after_load state: {exp_avg}")


def load_checkpoint_nodist():
    metadata = dist_cp.FileSystemReader(CHECKPOINT_DIR).read_metadata()

    state_dict = {}
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
        else:
            value: TensorStorageMetadata
            state_dict[key] = alloc_tensor(value.properties, value.size)

    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        no_dist=True
    )
    return unflatten_state_dict(state_dict, metadata.planner_data)

def no_dist_explore_state_dict():
    checkpoint = load_checkpoint_nodist()
    """
    When looking at the output of the sharded state note the following:

    net1.weight is sharded across FSDP groups
    net1.bias is replicated across FSDP groups

    FSDP groups are [0, 2] [1, 3] so the data is split in the following way:

    notation: [index: length]
    group0: weight [0: 8] bias [0: 4]
        rank 0: weight [0: 6]
        rank 2: weight [6: 2] bias [0: 4]
    group0: weight [8: 8] bias [0: 4]
        rank 0: weight [8: 6]
        rank 2: weight [14: 2] bias [0: 4]
    """
    print(checkpoint["state"]["net1.weight"]["exp_avg"])
    print(checkpoint["state"]["net1.bias"]["exp_avg"])

def work():
    # save_2d_model()
    # load_2d_model()

    save_2d_optim()
    # if dist.get_rank() == 0:
    #     no_dist_explore_state_dict()
    load_2d_optim()

if __name__ == "__main__":
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    dist_run(work, world_size=4)
