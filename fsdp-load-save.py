import shutil
from torch.distributed._shard.api import _shard_tensor
from torch.distributed._shard.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import torch.distributed._shard.checkpoint as dist_cp

from distcp_playground.utils import (
    traverse_state_dict,
    print_visitor,
)

from distcp_playground.nested import (
    RenameLoader,
    RenameSaver,
    unflatten_state_dict
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

def p0(line):
    if dist.get_rank() == 0:
        print(line)

def init_model():
    model = torch.nn.Linear(4, 8).cuda()
    model = FSDP(model)

    optim_input = list(model.parameters())
    optim = torch.optim.Adam(optim_input, lr=0.0001)

    return model, optim, optim_input

def save_sharded_model():
    torch.manual_seed(101)
    model, optim, optim_params = init_model()

    model(torch.rand(4).cuda()).sum().backward()
    optim.step()

    with FSDP.summon_full_params(model):
        p0(f"before-save: {model.weight}")
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model.state_dict()

        dist_cp.save_state_dict(
            state_dict=checkpoint,
            storage_writer=dist_cp.FileSystemWriter("checkpoint")
        )

def load_sharded_model():
    torch.manual_seed(103)
    model, optim, optim_params = init_model()
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model.state_dict()
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=dist_cp.FileSystemReader("checkpoint")
        )
        model.load_state_dict(checkpoint)
    with FSDP.summon_full_params(model):
        p0(f"after-load: {model.weight}")

def save_sharded_optim():
    torch.manual_seed(101)
    model, optim, optim_params = init_model()

    model(torch.rand(4).cuda()).sum().backward()
    optim.step()
    p0(f"before-save state: {optim.state_dict()}")

    optim_state = FSDP.sharded_optim_state_dict(model, optim, optim_params)
 
    dist_cp.save_state_dict(
        state_dict=optim_state,
        storage_writer=dist_cp.FileSystemWriter("checkpoint"),
        planner=RenameSaver()
    )

def alloc_tensor(props: TensorProperties, size: torch.Size):
    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device=torch.cuda.current_device()
    )

def key_stars_with(key, prefixes):
    return any(key.startswith(p) for p in prefixes)

def load_optimizer_state_dict():
    prefix = [ "state", "param_groups" ]
    metadata = dist_cp.FileSystemReader("checkpoint").read_metadata()
    sharding_spec = ChunkShardingSpec(
        dim=0,
        placements=[f"rank:{i}/cuda:{i}" for i in range(dist.get_world_size())]
    )

    state_dict = {}
    for key, value in metadata.state_dict_metadata.items():
        if not key_stars_with(key, prefix):
            continue
        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
        else:
            value: TensorStorageMetadata
            if len(value.chunks) == 1:
                state_dict[key] = alloc_tensor(value.properties, value.size)
            else:
                state_dict[key] = _shard_tensor(alloc_tensor(value.properties, value.size), sharding_spec)

    # Whether we unflatten before or after doesn't matter
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader("checkpoint")
    )
    state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

    if dist.get_rank() == 0:
        print("optimizer sharded state dict")
        traverse_state_dict(state_dict, print_visitor)
    return state_dict

def load_sharded_optim():
    torch.manual_seed(103)
    model, optim, optim_params = init_model()

    optim_state = load_optimizer_state_dict()

    flattened_osd = FSDP.flatten_sharded_optim_state_dict(
        optim_state, model, optim_params,
    )

    optim.load_state_dict(flattened_osd)
    p0(f"after-load state: {optim.state_dict()}")

def work():

    # save_sharded_model()
    # load_sharded_model()

    save_sharded_optim()
    load_sharded_optim()


if __name__ == "__main__":
    shutil.rmtree("checkpoint", ignore_errors=True)
    dist_run(work, world_size=2)
