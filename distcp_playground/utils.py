import torch

from typing import Mapping
from torch.distributed._shard.checkpoint.metadata import (
    STATE_DICT_TYPE,
)


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

def print_visitor(path, value, prefix=""):
    print(f"{prefix}[{path}] :: {type(value)}")
