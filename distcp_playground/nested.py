import io
import dataclasses
from dataclasses import dataclass
from functools import partial, reduce
from typing import Dict, Any, List, Tuple

import torch
import torch.distributed as dist

from torch.distributed._shard.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed._shard.checkpoint.planner import ReadItem, SavePlan
from torch.distributed._shard.checkpoint.utils import find_tensor_shard
from torch.distributed._shard.checkpoint.metadata import (
    STATE_DICT_TYPE,
    Metadata,
    MetadataIndex
)

from .utils import traverse_state_dict

def set_element(root_dict, path, value):
    cur_container = root_dict
    #populate
    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        if type(key) == str:
            cur_container = cur_container.setdefault(prev_key, {})
        else:
            cur_container = cur_container.setdefault(prev_key, [])

    key = path[-1]
    if type(key) == int:
        while len(cur_container) <= key:
            cur_container.append(None)
    cur_container[key] = value

def get_element(root_dict, path, default_value=None):
    cur_value = root_dict
    for part in path:
        if not part in cur_value:
            return default_value
        cur_value = cur_value[part]
    return cur_value

def print_visitor(path, value, prefix=""):
    print(f"{prefix}[{path}] :: {type(value)}")

def structural_copy(new_dict, path, obj):
    set_element(new_dict, path, obj)


def flatten_state_dict(state_dict: STATE_DICT_TYPE) -> Tuple[STATE_DICT_TYPE, dict[str, Tuple]]:
    flattened = {}
    mappings = {}
    def flat_copy(path, value):
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings

def unflatten_state_dict(state_dict: STATE_DICT_TYPE, mapping: dict[str, Tuple]) -> STATE_DICT_TYPE:
    inflated = {}
    for key, value in state_dict.items():
        set_element(inflated, mapping[key], value)
    return inflated 


class RenameSaver(DefaultSavePlanner):
    """
    Example SavePlanner that uses flatten_state_dict to handle complex state_dict objects
    """
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        super().init(state_dict, is_coordinator)
        self.state_dict, self.mappings = flatten_state_dict(state_dict)

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        return dataclasses.replace(plan, planner_data=self.mappings)

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        global_plan, metadata = super().create_global_plan(all_plans)
        merged_mappings = reduce(lambda x, y: x | y, (p.planner_data for p in global_plan))
        metadata = dataclasses.replace(metadata, planner_data=merged_mappings)
        return global_plan, metadata

class RenameLoader(DefaultLoadPlanner):
    """
    Example LoadPlanner that uses flatten_state_dict to handle complex state_dict objects
    """
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        super().init(state_dict, metadata, is_coordinator)
        self.original_state_dict = state_dict
        self.state_dict, self.mappings = flatten_state_dict(state_dict)
        self.checkpoint_mappings = metadata.planner_data
        if dist.get_rank() == 0:
            traverse_state_dict(self.state_dict, partial(print_visitor, prefix="load-time"))

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        set_element(self.original_state_dict, self.checkpoint_mappings[read_item.dest_index.fqn], torch.load(value))

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        obj = get_element(self.original_state_dict, self.mappings[index.fqn])
        return find_tensor_shard(obj, index)
