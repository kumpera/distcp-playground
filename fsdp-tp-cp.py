import shutil
import os
from typing import Tuple

import torch
import torch.distributed as dist

from torch.distributed._shard import shard_parameter
from torch.distributed._shard.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec

from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP



import torch.distributed._shard.checkpoint as dist_cp

from distcp_playground.utils import (
    traverse_state_dict,
    print_visitor,
)

from distcp_playground.dist_2d import (
    NestedTensorSaver,
    NestedTensorLoader
)

from distcp_playground.run import dist_run

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

def p0(line):
    if dist.get_rank() == 0:
        print(line)

def init_model():
    model = torch.nn.Linear(4, 8).cuda()
    model = FSDP(model)

    optim_input = list(model.parameters())
    optim = torch.optim.Adam(optim_input, lr=0.0001)

    return model, optim, optim_input

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
        f"rank:{idx}/cuda:{dist.distributed_c10d._get_global_rank(pg, idx) % torch.cuda.device_count()}"
        for idx in range(pg.size())
    ]
    return ChunkShardingSpec(
        dim=0,
        placements=placements,
    )

def save_2d_model():
    torch.manual_seed(101)
    tp_pg, dp_pg = create_2d_process_groups(TP_DEGREE)

    model_tp = torch.nn.Linear(4, 4).cuda()
    sharding_spec = create_colwise_spec(tp_pg)
    shard_parameter(model_tp, "weight", sharding_spec, process_group=tp_pg)
    model_tp = FSDP(model_tp, process_group=dp_pg)

    with FSDP.summon_full_params(model_tp):
        print(f"{dist.get_rank()} :: before-save: {model_tp.weight.local_tensor()}")

    dist.barrier()

    with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model_tp.state_dict()
        dist_cp.save_state_dict(
            state_dict=checkpoint,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=NestedTensorSaver())

def load_2d_model():
    torch.manual_seed(101)
    tp_pg, dp_pg = create_2d_process_groups(TP_DEGREE)

    model_tp = torch.nn.Linear(4, 4).cuda()
    sharding_spec = create_colwise_spec(tp_pg)
    shard_parameter(model_tp, "weight", sharding_spec, process_group=tp_pg)
    model_tp = FSDP(model_tp, process_group=dp_pg)


    dist.barrier()

    with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model_tp.state_dict()
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=dist_cp.FileSystemReader(path=CHECKPOINT_DIR),
            planner=NestedTensorLoader())

    with FSDP.summon_full_params(model_tp):
        print(f"{dist.get_rank()} :: after-load: {model_tp.weight.local_tensor()}")

def work():
    save_2d_model()
    load_2d_model()

    # save_sharded_optim()
    # load_sharded_optim()


if __name__ == "__main__":
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    dist_run(work, world_size=4)

# class ShardedTensor:
#     pass
# class Shard:
#     pass

# tensor =[]
# dp_tp = 1
# tp_pg= 1
# global_process_group=1
# # {
# #     "weights": ShardedTensor(
# #         Shard("rank0", tensor[4]),
# #         Shard("rank1", tensor[4]),
# #     ),
# #     "bias": ShardedTensor(
# #         Shard("rank0", tensor[2]),
# #         Shard("rank1", tensor[2]),
# #     ),
# # },

# {
#     "weights": ShardedTensor(
#         Shard("fpdp0-rank0", tensor[2]),
#         Shard("fpdp0-rank1", tensor[2]),
#         Shard("fpdp1-rank0", tensor[2]),
#         Shard("fpdp1-rank1", tensor[2]),
#     ),
# }