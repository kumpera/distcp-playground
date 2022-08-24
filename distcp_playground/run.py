# -*- coding: utf-8 -*-

import logging
import os
import traceback
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from torch.testing._internal.common_distributed import tp_transports


def init_pg(rank, world_size):
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        init_method = "env://"
    )

    # set device for nccl pg for collectives
    torch.cuda.set_device(rank)


def init_rpc(rank, world_size):
    # This is a workaround for EFA and TensorPipe
    # options = dist.rpc.TensorPipeRpcBackendOptions(_transports=tp_transports())
    # We're lazy, this is faster to init and produces less console spam
    options = dist.rpc.TensorPipeRpcBackendOptions(_transports=["shm", "uv"])
    # if torch.cuda.is_available():
    #     for i in range(world_size):
    #         if i != rank:
    #             options.set_device_map(f"worker{i}", { rank: i })

    dist.rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options)

def init_comms(options, rank, world_size):
    if options["c10d"]:
        init_pg(rank, world_size=world_size)
    if options["rpc"]:
        init_rpc(rank, world_size)

def destroy_comms(options):
    if options["c10d"]:
        dist.barrier()
        dist.destroy_process_group()
    if options["rpc"]:
        dist.rpc.shutdown()

def worker(rank, run_fn, options, world_size):
    logging.basicConfig(filename=f"rank{rank}", level=logging.WARNING, force=True)
    init_comms(options, rank, world_size=world_size)

    try:
        run_fn()
    except:
        traceback.print_exc()

    destroy_comms(options)

# TODO support Gloo
def dist_run(run_fn, world_size=0, init_rpc=False, init_c10d=True):
    world_size = world_size or torch.cuda.device_count()

    options = {
        "rpc": init_rpc,
        "c10d": init_c10d
    }

    port = random.randint(10000, 20000)
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] = str(port)
    # init_method = f"tcp://localhost:{port}"
    # options["init_method"] = init_method


    mp.spawn(
        fn=worker,
        args=(run_fn, options, world_size, ),
        nprocs=world_size,
        join=True,
    )

