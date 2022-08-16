import traceback
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import torch

def init_pg(init_method, rank, world_size):
    import torch

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        init_method = init_method
    )

    # set device for nccl pg for collectives
    torch.cuda.set_device(rank)


def init_comms(file_name, rank, world_size):
    init_pg(file_name, rank, world_size=world_size)

def destroy_comms():
    dist.barrier()
    dist.destroy_process_group()

def worker(rank, run_fn, init_method, world_size):
    init_comms(init_method, rank, world_size=world_size)

    try:
        run_fn()
    except:
        traceback.print_exc()

    destroy_comms()

# TODO support Gloo
def dist_run(run_fn, world_size=0):
    port = random.randint(10000, 20000)
    init_method = f"tcp://localhost:{port}"
    world_size = world_size or torch.cuda.device_count()

    mp.spawn(
        fn=worker,
        args=(run_fn, init_method, world_size, ),
        nprocs=world_size,
        join=True,
    )

