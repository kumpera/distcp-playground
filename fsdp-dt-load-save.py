import os
import shutil
import functools


from torch.distributed._shard.api import _shard_tensor
from torch.distributed._shard.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from spmd import distribute_tensor, DeviceMesh, DTensor as DT, Shard, Replicate

import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed.distributed_c10d as distributed_c10d

from distcp_playground.utils import (
    traverse_state_dict,
    print_visitor,
    print_sharded_tensor,
)

from distcp_playground.dist_2d import (
    NestedTensorLoader,
    NestedDedupTensorSaver,
)

from distcp_playground.run import dist_run

# Tensor-Parallel degree
TP_DEGREE = 2
CHECKPOINT_DIR = f"/scratch/{os.environ['LOGNAME']}/checkpoint"
LR = 3e-5

OPS_NOT_SHARD = [
    "net3.weight",
    "net3.bias",
]

SHARD_PARAMS = [
    "net1.weight",
    "net1.bias",
    "net2.weight",
]

def p0(line):
    if dist.get_rank() == 0:
        print(line)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, self.net1.weight.size(1))
        result = self.net2(self.relu(self.net1(x)))
        return self.net3(result.view(*x_size[:-1], -1))


def _get_module_optim(module):
    return torch.optim.SGD(module.parameters(), lr=LR)

def _aggregate_local_tensor(module: torch.nn.Module) -> torch.nn.Module:
    def hook_func(_module, _input, output):
        if isinstance(output, DT):
            replica_placement = [Replicate()]
            return output.redistribute(
                output.device_mesh, replica_placement
            ).to_local()

    module.register_forward_hook(hook_func)
    return module


def _replicate_input_tensor(
    module: torch.nn.Module, device_mesh, replica_placement
) -> torch.nn.Module:
    def hook_func(_, input):
        if not isinstance(input[0], DT):
            return DT(input[0], device_mesh, replica_placement)

    module.register_forward_pre_hook(hook_func)
    return module


def _gradient_hook(param, grad):
    print("grad._local_tensor", grad._local_tensor)
    param._local_tensor.grad = grad._local_tensor


def shard_module(m, pg):
    start_idx = distributed_c10d._get_global_rank(pg, 0)
    device_mesh = DeviceMesh(
        "cuda", list(range(start_idx, start_idx + pg.size())), dim_groups=[pg]
    )
    col_wise_sharding = [Shard(0)]
    row_wise_sharding = [Shard(1)]
    replicate = [Replicate()]
    m.net1.weight = torch.nn.Parameter(
        distribute_tensor(m.net1.weight, device_mesh, col_wise_sharding),
    )
    m.net2.weight = torch.nn.Parameter(
        distribute_tensor(m.net2.weight, device_mesh, row_wise_sharding)
    )
    m.net1.bias = torch.nn.Parameter(
        distribute_tensor(m.net1.bias, device_mesh, col_wise_sharding)
    )
    m.net2.bias = torch.nn.Parameter(
        distribute_tensor(m.net2.bias, device_mesh, replicate)
    )
    m = _replicate_input_tensor(m, device_mesh, replicate)
    m.net2 = _aggregate_local_tensor(m.net2)
    m.net1.weight.register_hook(
        functools.partial(_gradient_hook, m.net1.weight)
    )
    m.net2.weight.register_hook(
        functools.partial(_gradient_hook, m.net2.weight)
    )
    m.net1.bias.register_hook(
        functools.partial(_gradient_hook, m.net1.bias)
    )
    m.net2.bias.register_hook(
        functools.partial(_gradient_hook, m.net2.bias)
    )


def _shard_wrap_module(module, module_shard, fsdp_wrap, tp_pg, fsdp_pg):
    if module_shard:
        # Fetch the module sharding planner.
        shard_module(module, tp_pg)

    if fsdp_wrap and module_shard:
        return FSDP(module, process_group=fsdp_pg)
    if fsdp_wrap:
        return FSDP(module, process_group=distributed_c10d._get_default_group())
    return module


def init_model():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()

    # Use same seed so that each rank get the same model params.
    # args = _get_args_for_script()
    model_parallel_size = TP_DEGREE
    model = SimpleModel().cuda(rank)
    model_optim = _get_module_optim(model)

    tp_ids = []
    fsdp_ids = []
    for i in range(world_size):
        idx = i // model_parallel_size
        if len(tp_ids) <= idx:
            tp_ids.append([])
        tp_ids[idx].append(i)
        idx = i % model_parallel_size
        if len(fsdp_ids) <= idx:
            fsdp_ids.append([])
        fsdp_ids[idx].append(i)

    tp_pgs = [dist.new_group(ids) for ids in tp_ids]
    data_parallel_pgs = [dist.new_group(ids) for ids in fsdp_ids]
    tp_pg = tp_pgs[rank // model_parallel_size]
    fsdp_pg = data_parallel_pgs[rank % model_parallel_size]

    # Create Input
    model = _shard_wrap_module(
        model, True, True, tp_pg, fsdp_pg
    )
    return model

def save_dt_model():
    torch.manual_seed(103)
    model = init_model()

    with FSDP.summon_full_params(model):
        p0(f"before-save: net1.bias {model.net1.bias}")
        p0(f"before-save: net2.bias {model.net2.bias}")

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model.state_dict()

        dist_cp.save_state_dict(
            state_dict=checkpoint,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=NestedDedupTensorSaver()
        )

def load_dt_model():
    dist.barrier()
    p0("-------------")
    dist.barrier()
    torch.manual_seed(101)
    model = init_model()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model.state_dict()

        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=NestedTensorLoader()
        )
        model.load_state_dict(checkpoint)

    with FSDP.summon_full_params(model):
        p0(f"after-load: net1.bias {model.net1.bias}")
        p0(f"after-load: net2.bias {model.net2.bias}")

def work():
    save_dt_model()
    load_dt_model()

    # save_sharded_optim()
    # load_sharded_optim()


if __name__ == "__main__":
    shutil.rmtree("checkpoint", ignore_errors=True)
    dist_run(work, world_size=4)
