# -*- coding: utf-8 -*-

import torch

from typing import Mapping
from torch.distributed._shard.checkpoint.metadata import (
    STATE_DICT_TYPE,
)
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from spmd import  DTensor as DT
import logging

def keep_visiting_tensors(value):
    return isinstance(value, torch.Tensor)


def traverse_state_dict(state_dict: STATE_DICT_TYPE, visitor, keep_traversing=keep_visiting_tensors):
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.

    Traversal is shortcuted when if finds a collection for which `keep_visiting_tensors` evaluates
    to false for all elements.

    By default, all collections with at least one ``torch.Tensor`` element are traversed.

    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """
    # a value is terminal if it has no other containers values inside it
    def _is_terminal(value):
        values = None
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path, value):
        if _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)

def print_visitor(path, value, prefix="", print_fun=print):
    print_fun(f"{prefix}[{path}] :: {type(value)}")

def print_tensor(value, padding="", prefix="", print_fun=print):
    if isinstance(value, ShardedTensor):
        print_fun(f"{padding}{prefix}ShardedTensor size {value.size()}")
        for shard in value.local_shards():
            print_tensor(shard.tensor, f"{padding}\t", f"{shard.metadata.shard_offsets} ", print_fun=print_fun)
    elif isinstance(value, DT):
        print_fun(f"{padding}{prefix}DistributedTensor size {value.size()}")
        # for shard in value.local_shards():
        print_tensor(value.local_tensor, f"{padding}\t", f"(offset ???) ", print_fun=print_fun)
    else:
        print_fun(f"{padding}{prefix}Tensor size {value.size()}")


def print_sharded_tensor(path, value, print_fun=print):
    if not isinstance(value, ShardedTensor) and not isinstance(value, DT):
        print_visitor(path, value, print_fun=print_fun)
    else:
        print_tensor(value, prefix=path, print_fun=print_fun)


logger = logging.getLogger("distcp-playground")
LOGGER_INIT=False
def get_logger():
    global LOGGER_INIT
    if not LOGGER_INIT:
        logger.addHandler(logging.FileHandler(f"checkpoints/{torch.distributed.get_rank()}.log", "w+"))
        LOGGER_INIT = True
    return logger
