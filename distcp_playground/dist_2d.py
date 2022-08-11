import copy
import dataclasses
from typing import Dict,Any, List
from torch.distributed._shard.sharded_tensor import shard

from torch.distributed._shard.sharded_tensor.api import ShardedTensor
import torch.distributed as dist

from torch.distributed._shard.checkpoint.metadata import Metadata, MetadataIndex, STATE_DICT_TYPE
from torch.distributed._shard.checkpoint.resharding import create_read_items, create_write_items
from torch.distributed._shard.checkpoint.utils import find_state_dict_object
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata, TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.remote_device import _remote_device

from torch.distributed._shard.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner
)


import torch

def element_wise_add(a: torch.Size, b: torch.Size) -> torch.Size:
    return torch.Size([i_a + i_b for i_a, i_b in zip(a,b)])

def element_wise_add2(a: List[int], b: List[int]) -> List[int]:
    return [i_a + i_b for i_a, i_b in zip(a,b)]

def flatten_sharded_tensors(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = type(state_dict)()
    for key, value in state_dict.items():
        if not isinstance(value, ShardedTensor):
            new_state_dict[key] = value
            continue
        shards = value.local_shards()
        if len(shards) != 1:
            raise ValueError("Cannot handle outer tensor with more than 1 shard")
        outer_shard = shards[0]

        inner_st = outer_shard.tensor
        if not isinstance(inner_st, ShardedTensor):
            new_state_dict[key] = value
            continue

        if len(inner_st.local_shards()) != 1:
            raise ValueError("Cannot handle inner tensor with more than 1 shard")
        inner_shard = inner_st.local_shards()[0]

        local_shards = [
            Shard(
                tensor=inner_shard.tensor,
                metadata=ShardMetadata(
                    shard_offsets=element_wise_add2(
                        outer_shard.metadata.shard_offsets, 
                        inner_shard.metadata.shard_offsets),
                    shard_sizes=inner_shard.metadata.shard_sizes,
                    placement=f"rank:{dist.get_rank()}/cuda:{torch.cuda.current_device()}"
                ))
        ]

        st_meta: ShardedTensorMetadata = copy.deepcopy(value.metadata())
        other_rank = 0 if dist.get_rank() > 0 else 1
        # Remove the outer ST shard the inner ST covers
        for i, shard_md in enumerate(st_meta.shards_metadata):
            if shard_md.shard_offsets == outer_shard.metadata.shard_offsets:
                st_meta.shards_metadata.pop(i)
                break

        # blame other rank for the other shards
        for shard_md in st_meta.shards_metadata:
            shard_md.placement=_remote_device(f"rank:{other_rank}/cuda:0")

        # Add other inner shards from the inner tensor
        for inner_md in inner_st.metadata().shards_metadata:
            if inner_md.shard_offsets != inner_shard.metadata.shard_offsets:
                st_meta.shards_metadata.append(ShardMetadata(
                    shard_offsets=element_wise_add2(
                        outer_shard.metadata.shard_offsets, 
                        inner_md.shard_offsets),
                    shard_sizes=inner_md.shard_sizes,
                    placement=f"rank:{other_rank}/cuda:0"
                ))
        
        #finally add this shard
        st_meta.shards_metadata.append(local_shards[0].metadata)
        
        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards=local_shards,
            sharded_tensor_metadata=st_meta,
        )
        new_state_dict[key] = st

    return new_state_dict

class NestedTensorSaver(DefaultSavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        self.original_state_dict = state_dict
        return super().init(flatten_sharded_tensors(state_dict), is_coordinator)

class NestedTensorLoader(DefaultLoadPlanner):
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        self.original_state_dict = state_dict
        return super().init(flatten_sharded_tensors(state_dict), metadata, is_coordinator)