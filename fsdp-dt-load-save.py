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
    NestedRenamingTensorSaver,
    NestedDedupRenamingTensorSaver,
    load_2d_optimizer_state_dict,
    get_data_parallel_process_group,
    UberLoadPlanner,
    UberSavePlanner
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

TP_PG = None
DP_PG = None

def init_model():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()

    model_parallel_size = TP_DEGREE
    model = SimpleModel().cuda(rank)

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
    global TP_PG
    global DP_PG
    TP_PG = tp_pg
    DP_PG = fsdp_pg

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

        state_dict = {
            "model": checkpoint,
            "other": "save-time-string"
        }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=UberSavePlanner()
        )

def load_dt_model():
    dist.barrier()
    p0("-------------")
    dist.barrier()
    torch.manual_seed(101)
    model = init_model()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model.state_dict()
        state_dict = {
            "model": checkpoint,
            "other": ""
        }
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=UberLoadPlanner()
        )
        model.load_state_dict(state_dict["model"])

    with FSDP.summon_full_params(model):
        p0(f"after-load: net1.bias {model.net1.bias}")
        p0(f"after-load: net2.bias {model.net2.bias}")

def save_dt_optim():
    torch.manual_seed(107)
    model_tp = init_model()
    optim_input = list(model_tp.parameters())
    optim = torch.optim.Adam(optim_input, lr=0.0001)

    model_tp(torch.rand(5).cuda()).sum().backward()
    optim.step()

    optim_state = FSDP.sharded_optim_state_dict(model_tp, optim, optim_input)
    net1_bias = optim_state["state"]["net1.bias"]["exp_avg"]
    net2_bias = optim_state["state"]["net2.bias"]["exp_avg"]
    net1_bias = net1_bias.local_tensor().local_tensor() 
    net2_bias = net2_bias.local_tensor().local_tensor()
    for i in range(dist.get_world_size()):
        if i == dist.get_rank(): 
            print(f"[[{dist.get_rank()}]] before-save optim-state: net1:{net1_bias} net2:{net2_bias}")
        dist.barrier()

    state_dict = {
        "optimizer_state": optim_state
    }

    md = dist_cp.save_state_dict(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        planner=NestedDedupRenamingTensorSaver()
    )

def dump_checkpoint():
    dist.barrier()
    if dist.get_rank() == 0:
        metadata = dist_cp.FileSystemReader(CHECKPOINT_DIR).read_metadata()
        load_keys = [ "optimizer_state.state.net1.bias.exp_avg", "optimizer_state.state.net2.bias.exp_avg" ]

        state_dict = {}
        for key in load_keys:
            md = metadata.state_dict_metadata[key]
            state_dict[key] = torch.zeros(md.size)

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            no_dist=True
        )

        for key, value in state_dict.items():
            print(f"{key} :: {value}")
    dist.barrier()

def load_dt_optim():
    torch.manual_seed(103)
    model_tp = init_model()
    optim_input = list(model_tp.parameters())
    optim = torch.optim.Adam(optim_input, lr=0.0001)


    the_pg = get_data_parallel_process_group(model_tp)

    assert the_pg == DP_PG

    with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model_tp.state_dict()
        optim_state = load_2d_optimizer_state_dict(
            model_state_dict,
            optimizer_key="optimizer_state",
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR), 
            dp_pg=DP_PG
        )

        net1_bias = optim_state["optimizer_state"]["state"]["net1.bias"]["exp_avg"]
        net2_bias = optim_state["optimizer_state"]["state"]["net2.bias"]["exp_avg"]
        net1_bias = net1_bias.local_tensor()
        net2_bias = net2_bias.local_tensor()
        for i in range(dist.get_world_size()):
            if i == dist.get_rank(): 
                print(f"[[{dist.get_rank()}]] after-load optim-state: net1:{net1_bias} net2:{net2_bias}")
            dist.barrier()

        flattened_osd = FSDP.flatten_sharded_optim_state_dict(
            optim_state["optimizer_state"], model_tp, optim_input, DP_PG
        )

        optim.load_state_dict(flattened_osd)


def work():
    # save_dt_model()
    # load_dt_model()

    save_dt_optim()
    load_dt_optim()


if __name__ == "__main__":
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    dist_run(work, world_size=4)
