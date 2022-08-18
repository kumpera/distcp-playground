import os
import math
import shutil
import time
from typing import Any, Tuple
from torch.distributed._shard.checkpoint.metadata import MetadataIndex

from torch.distributed._shard.checkpoint.planner import SavePlan
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from torch.nn import parameter

from distcp_playground.run import dist_run
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._shard._utils import narrow_tensor_by_index

import torch.distributed._shard.checkpoint as dist_cp

from distcp_playground.utils import (
    traverse_state_dict,
    print_visitor,
)

from distcp_playground.spaces import RemoteTensor, Spaces

CHECKPOINT_DIR = f"/scratch/{os.environ['LOGNAME']}/checkpoint"



def _get_flat_param_coords(
    numel: int,
    rank: int,
    world_size: int,
) -> int:
    """Returns (offset, length) of a the local flatparam of `rank` given a `world_size` split"""

    tensor = torch.empty(numel, device="meta")
    chunks = torch.flatten(tensor).chunk(world_size)
    if len(chunks) < (rank + 1):
        return -1

    rank_offset = 0
    for i in range(0, rank):
        rank_offset += chunks[i].numel()
    return rank_offset, chunks[rank].numel()
   
def fsdp_remote_tensor_state_dict(model, space):
    flat_p = list(model.parameters())[0]
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_start, local_len = _get_flat_param_coords(flat_p._unsharded_size, local_rank, world_size)
    local_end = local_start + local_len

    tensor_start = 0

    to_cpu_time = 0
    register_time = 0

    state_dict = {}
    for pinfo, numel, shape, name in zip(flat_p._param_infos, flat_p._numels, flat_p._shapes, flat_p._prefixed_param_names):
        tensor_end = tensor_start + numel
        if not (tensor_end < local_start or tensor_start > local_end):
            tensor_local_offset = max(0, local_start - tensor_start)
            tensor_local_start = max(0, tensor_start - local_start)
            tensor_local_len = min(local_end, tensor_end) - max(local_start, tensor_start) 

            #FIXME this is stupid, but TensorPipe is annoying AF when doing CUDA
            start = time.time()
            local_tensor = torch.narrow(flat_p, 0, tensor_local_start, tensor_local_len)
            if tensor_local_len < math.prod(shape):
                local_tensor = local_tensor.cpu()
            to_cpu_time += time.time() - start

            start = time.time()
            space.register_linear(name, shape, tensor_local_offset, tensor_local_len, local_tensor)
            register_time += time.time() - start

        state_dict[name] = space.get_tensor(name)

        tensor_start += numel
    barrier_time = time.time()
    dist.barrier()
    barrier_time = time.time() - barrier_time
    p0(f"to-cpu: {to_cpu_time}s register:{register_time}s barrier:{barrier_time}s")

    return state_dict

def work():
    id = rpc.get_worker_info().id
    print(f"hello from {id}")

    space = Spaces()
    torch.rand(1919 + id)
    my_tensor = torch.rand(4)
    print(f"{id} private data: {my_tensor}")
    dist.barrier()

    # register a tensor and a shard to a tensor
    space.register(f"tensor_{id}", my_tensor)
    space.register_shard(name=f"sharded", global_size=[4], offset=[id * 2], length=[2], local_tensor=my_tensor[0:2])
    if id == 0:
        space.register_linear(name=f"linear", global_size=[2,2], offset=0, length=3, local_tensor=my_tensor[0:3])
    else:
        space.register_linear(name=f"linear", global_size=[2,2], offset=3, length=1, local_tensor=my_tensor[0:1])

    dist.barrier()
    t0 = space.get_tensor("tensor_0")
    print(f"I'm {id} and t0 is {t0.local_copy()} localfyness: {t0.get_localfyness()}")

    t1 = space.get_tensor("tensor_1")
    print(f"I'm {id} and t1 is {t1.local_copy()} localfyness: {t1.get_localfyness()}")

    sharded = space.get_tensor("sharded")
    print(f"I'm {id} and sharded is {sharded.local_copy()} localfyness: {sharded.get_localfyness()}")

    linear = space.get_tensor("linear")
    print(f"I'm {id} and linear is {linear.local_copy()} localfyness: {linear.get_localfyness()}")


def p0(line):
    if dist.get_rank() == 0:
        print(line)
 

def print_localfyness(path, value):
    if isinstance(value, RemoteTensor):
        print(f"({dist.get_rank()}) [{path}] :: remote-tensor: {value.get_localfyness() * 100.0}% local")
    else:
        print(f"({dist.get_rank()}) [{path}] :: local-value: {type(value)}")

class MyModel(torch.nn.Module):

    def __init__(self, inner_module=None):
        super(MyModel, self).__init__()

        self.seq = torch.nn.Sequential(*[torch.nn.Linear(10_000, 10_000, device="meta") for _ in range (20)])
        self.net1 = torch.nn.Linear(10_1000, 10, device="meta")

    def reset_parameters(self):
        pass

def big_model():
    return FSDP(MyModel())


class RemoteTensorPlanner(DefaultSavePlanner):
    def create_local_plan(self) -> SavePlan:
        requests = []
        self.remote_tensors = {}
        for fqn, obj in self.state_dict.items():
            if isinstance(obj, RemoteTensor):
                # FIXME we should use a rule that is unambiguous if a tensor is split half and half
                if obj.get_localfyness() >= 0.5:
                    # print(f"({dist.get_rank()}) => gonna save tensor {fqn} with localfyness {obj.get_localfyness()}")
                    tensor = obj.local_copy()
                    reqs = dist_cp.resharding.create_write_items(fqn, tensor)
                    assert len(reqs) == 1
                    self.remote_tensors[reqs[0].index] = tensor
                    requests += reqs
            elif isinstance(obj, ShardedTensor) or self.is_coordinator:
                requests += dist_cp.resharding.create_write_items(fqn, obj)
        return SavePlan(requests)

    def lookup_object(self, index: MetadataIndex) -> Any:
        if index in self.remote_tensors:
            return self.remote_tensors[index]
        return super().lookup_object(index)

def fsdp_integration():
    # model = FSDP(torch.nn.Linear(4, 5).cuda())
    model = big_model()
    space = Spaces()
    dist.barrier()
    do_sharded_save = False

    if do_sharded_save:
        state_dict_time = time.time()
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            checkpoint = model.state_dict()
            state_dict_time = time.time() - state_dict_time

        save_time = time.time()
        dist_cp.save_state_dict(
            state_dict=checkpoint,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR)
        )
        save_time = time.time() - save_time

    else:
        state_dict_time = time.time()
        state_dict = fsdp_remote_tensor_state_dict(model, space)
        state_dict_time = time.time() - state_dict_time

        save_time = time.time()
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=RemoteTensorPlanner()
        )
        save_time = time.time() - save_time

    p0(f"state_dict creation took {state_dict_time}s save took: {save_time}s using FSDP sharded:{do_sharded_save}")
    
    """
    A quick benchmark of this toy sample on 1 host with 8 GPUs:

    to-cpu: 1.0948028564453125s register:0.00415492057800293s barrier:0.022481441497802734s
    state_dict creation took 1.1220619678497314s save took: 6.1418633460998535s using FSDP sharded:False

    only move to cpu() if tensor is sharded
    to-cpu: 0.21965289115905762s register:0.0033528804779052734s barrier:0.44196248054504395s
    state_dict creation took 0.6655890941619873s save took: 5.911475658416748s using FSDP sharded:False

    state_dict creation took 0.17703866958618164s save took: 5.467177867889404s using FSDP sharded:True

    Few things to notice:
        TensorPipe costs us the to-cpu time - I could not get the device map thing to work.
        Barrier is expensive cuz it hides some tail to-cpu cost.
            Try per-tensor lazy sync to increase parallelism
            Add to_local_async and use it to hide the network cost
        
        I wonder why save is faster with FSDP. Theories:
            flatparam is batch moved to CPU?

        TLDR: we can't compete with NVLink, measure this with 8 hosts and 1 GPU

    """



if __name__ == "__main__":
    # dist_run(work, world_size=2, init_rpc=True, init_c10d=True)
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    dist_run(fsdp_integration, world_size=8, init_rpc=True, init_c10d=True)