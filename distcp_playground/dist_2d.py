import copy
import dataclasses
from functools import reduce, partial
import io
import math
from pickle import FALSE
from typing import Dict,Any, List, Sequence, Tuple
from torch.distributed._shard.checkpoint.planner import LoadPlan, ReadItem, SavePlan
from torch.distributed._shard.sharded_tensor import shard

from torch.distributed._shard.sharded_tensor.api import ShardedTensor
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint.metadata import BytesStorageMetadata, ChunkStorageMetadata, Metadata, MetadataIndex, STATE_DICT_TYPE, TensorStorageMetadata
from torch.distributed._shard.checkpoint.planner_helpers import _create_sharded_read_items, _create_read_items
from torch.distributed._shard.checkpoint.utils import find_state_dict_object, find_tensor_shard
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata, TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.remote_device import _remote_device

from spmd import DTensor as DT
from torch.distributed._shard.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner
)
from torch.distributed._shard.api import _shard_tensor

import distcp_playground.nested as cp_nested

from distcp_playground.utils import (
    traverse_state_dict,
    print_sharded_tensor,
    get_logger
)

import torch

def element_wise_add(a: List[int], b: List[int]) -> List[int]:
    return [i_a + i_b for i_a, i_b in zip(a,b)]

def flatten_sharded_tensors(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}

    def rewrite_dict(path, value):
        if not isinstance(value, ShardedTensor):
            cp_nested.set_element(new_state_dict, path, value)
            return
        shards = value.local_shards()
        if len(shards) == 0:
            return
        if len(shards) != 1:
            raise ValueError(f"Cannot handle outer tensor with more than 1 shard {path} -- {len(shards)}")
        outer_shard = shards[0]

        inner_st = outer_shard.tensor
        if not isinstance(inner_st, ShardedTensor):
            cp_nested.set_element(new_state_dict, path, value)
            return

        if len(inner_st.local_shards()) != 1:
            raise ValueError("Cannot handle inner tensor with more than 1 shard")
        inner_shard = inner_st.local_shards()[0]

        local_shards = [
            Shard(
                tensor=inner_shard.tensor,
                metadata=ShardMetadata(
                    shard_offsets=element_wise_add(
                        outer_shard.metadata.shard_offsets, 
                        inner_shard.metadata.shard_offsets),
                    shard_sizes=inner_shard.metadata.shard_sizes,
                    placement=f"rank:{dist.get_rank()}/{inner_shard.tensor.device}"
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
                    shard_offsets=element_wise_add(
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
        cp_nested.set_element(new_state_dict, path, st)

    traverse_state_dict(state_dict, rewrite_dict)
    return new_state_dict

def dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    all_plans = list(all_plans)
    key_to_plan = {}
    for plan_idx, plan in enumerate(all_plans):
        for wi in plan.items:
            key_to_plan.setdefault(wi.index, []).append(plan_idx)

    replicated_items = { k: v for k,v in key_to_plan.items() if len(v) > 1}

    # We're now presented with an interesting choice, how to remove duplicates
    # We can 1) always keep first entry; 2)randomly keep one entry; 3) load balance across rank
    # For now we do (1)
    # Compute the per-rank remove set
    plan_to_keys = {}
    for key, plans in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)

    for plan_idx, keys in plan_to_keys.items():
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [wi for wi in all_plans[plan_idx].items if wi.index not in key_set]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)
    return all_plans

def is_nested_tensor(val: Any) -> bool:
    if isinstance(val, ShardedTensor):
        if len(val.local_shards()) == 0:
            return False
        if isinstance(val.local_shards()[0].tensor, ShardedTensor):
            return True
        if isinstance(val.local_shards()[0].tensor, DT):
            raise ValueError("Cannot handle DT nested insided ST")
    # Safety valve for when this eventually happen
    elif isinstance(val, DT) and isinstance(val._local_tensor, (DT, ShardedTensor)):
        raise ValueError("Cannot handle nested DT")
    return False


def get_state_dict_2d_layout(state_dict: STATE_DICT_TYPE) -> Dict[str, Tuple[Sequence[int], Sequence[int]]]:
    """
    We have to load the right TP slice of the optimizer state.
    This is not easy since the per-tensor slicing can't be inferred from checkpoint metadata.
    We take advantage of the model state_dict producing a sliced ST to figure out what we need to load.
    This is pretty fragile and it might be easier for FSDP to compute this info for us.

    Returns a dictionary where keys are the same of the state_dict and the value is a tuple of
    (offset, size) for the current rank TP slice.

    N.B. The state_dict *MUST* come from FSDP.sharded_state_dict.
    """
    specs = {}
    for key, value in state_dict.items():
        specs[key] = (None, value.size())
        if is_nested_tensor(value):
            assert len(value.local_shards()) == 1, "Cannot handle ST with multiple shards"
            shard = value.local_shards()[0]
            specs[key] = (shard.metadata.shard_offsets, shard.metadata.shard_sizes)

    return specs

def gen_rank_device(global_rank):
    if torch.cuda.is_available():
        return f"cuda:{global_rank % torch.cuda.device_count()}"
    return "cpu"

def create_colwise_spec(pg):
    placements = [
        f"rank:{idx}/{gen_rank_device(dist.distributed_c10d._get_global_rank(pg, idx))}"
        for idx in range(pg.size())
    ]
    return ChunkShardingSpec(
        dim=0,
        placements=placements,
    )

# Same issue as 
def get_cur_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return "cpu"

def alloc_tensor(props: TensorProperties, size: torch.Size):
    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device=torch.cuda.current_device()
    )


def pg_to_str(pg: dist.ProcessGroup) -> str:
    if pg == dist.distributed_c10d._get_default_group():
        return "global"

    # looking into internals more than Jack the Ripper
    global_ranks = list(dist.distributed_c10d._pg_group_ranks[pg].keys())
    pg_membership = "_".join(str(x) for x in global_ranks)

    return pg_membership

def get_data_parallel_process_group(model: torch.nn.Module):
    from torch.distributed.fsdp._optim_utils import _get_flat_param_to_fsdp_module

    mods = _get_flat_param_to_fsdp_module(model)
    the_pg = None
    default_pg = dist.distributed_c10d._get_default_group()
    broken = False
    last_str = None
    for module in mods.values():
        other_pg = module.process_group
        if other_pg is not None:
            if the_pg is not None and module.process_group != the_pg:
                last_str = f"Mismatch DP groups among FSDP instances. Found {pg_to_str(the_pg)} and {pg_to_str(other_pg)}"
                get_logger().warning(last_str)
                broken = True
            else:
                the_pg = module.process_group

    if broken:
        raise ValueError(last_str)
    return the_pg

# TODO enable device control (right now we use CUDA whenever possible)
# Picking a device is not easy as we need to know the PG backend capabilities
# And cuda is even worse cuz we have to hardcode whether we're using CUDA_VISIBLE_DEVICES or not
# To sum up, the ST API is effing hard to work with and most of the data we pass in is useless
def load_2d_optimizer_state_dict(model_state_dict, optimizer_key, storage_reader, dp_pg):
    metadata = storage_reader.read_metadata()

    layout_specs = get_state_dict_2d_layout(model_state_dict)
    tp_spec = create_colwise_spec(dp_pg)
    # Create a state_dict for optimizer state
    state_dict = {}
    """
    TODO this depends on flatten MD, we're better using path objects here
    as they cleanly solve the prefix/suffix
    FSDP generates a 2 layer dict:
        "state": {
            "param_name": {
                "optimizer_param_0"
            }
        }
    
    If we assume that the whole optimizer state is put under a single key, say 'optim', we'd be able to encode this as follows:

    get all keys with path prefix: ('optim')

    For all tensor types we use the 3rd component as the ke into spec_key, IE:
    ('optim', 'state', 'net1.bias', 'exp_avg') -> 'net1.bias'

    """
    fqn_to_offset = {}
    for key, value in metadata.state_dict_metadata.items():
        key_path = metadata.planner_data[key]
        if key_path[0] != optimizer_key:
            continue

        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
            continue
        value: TensorStorageMetadata
        if value.size.numel() == 1:
            state_dict[key] = alloc_tensor(value.properties, value.size)
        else:
            spec_key = key_path[2]
            alloc_size = layout_specs.get(spec_key, (None, value.size))[1]

            st_md = tp_spec.build_metadata(alloc_size, value.properties)
            local_shards = []
            current_rank = dist.get_rank(dp_pg)
            for shard_md in st_md.shards_metadata:
                if shard_md.placement.rank() != current_rank:
                    continue
                local_shards.append(
                    Shard(
                        tensor=alloc_tensor(value.properties, shard_md.shard_sizes),
                        metadata=shard_md,
                    )
                )

            st = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )

            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = layout_specs[spec_key][0]

            state_dict[key] = st

    # Whether we unflatten before or after doesn't matter
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=ReaderWithOffset(fqn_to_offset)
    )

    state_dict = cp_nested.unflatten_state_dict(state_dict, metadata.planner_data)

    return state_dict


def check_box_overlap(box0, box1):
    """
    Checks if two boxes overlap. Tuples are (offset, lengths)
    """

    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(box0.offsets)
    for i in range(ndims):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False

    return True

def check_box_bounds(outer_box_size, inner_box):
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False

    return True

def validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> bool:
    all_good = True
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        #check for overlap
        chunks_volume = 0
        for chunk_idx, chunk0 in enumerate(value.chunks):
            if not check_box_bounds(value.size, chunk0):
                get_logger().warning(f"key:{key} has out of bounds chunk: tensor-size:{value.size} chunk: {chunk0}")
                all_good = False
            chunks_volume += math.prod(chunk0.sizes)

            for chunk1 in value.chunks[chunk_idx + 1:]:
                if check_box_overlap(chunk0, chunk1):
                    get_logger().warning(f"key:{key} has overlapping chunks: {chunk0} {chunk1}")
                    all_good = False

        tensor_volume = math.prod(value.size)
        if chunks_volume != tensor_volume:
            get_logger().warning(f"key:{key} invalid fill tensor-volume: {tensor_volume} chunks-volume: {chunks_volume}")
            all_good = False

    return all_good

class NestedTensorSaver(DefaultSavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        return super().init(flatten_sharded_tensors(state_dict), is_coordinator)

class NestedTensorLoader(DefaultLoadPlanner):
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        return super().init(flatten_sharded_tensors(state_dict), metadata, is_coordinator)

class NestedRenamingTensorSaver(DefaultSavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        state_dict, self.mappings = cp_nested.flatten_state_dict(state_dict)
        state_dict = flatten_sharded_tensors(state_dict)
        return super().init(state_dict, is_coordinator)

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        return dataclasses.replace(plan, planner_data=self.mappings)

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        global_plan, metadata = super().create_global_plan(all_plans)
        merged_mappings = reduce(lambda x, y: x | y, (p.planner_data for p in global_plan))
        metadata = dataclasses.replace(metadata, planner_data=merged_mappings)
        return global_plan, metadata


def print_to_log(log_line):
    get_logger().info(f"{dist.get_rank()} :: {log_line}")

class NestedDedupTensorSaver(DefaultSavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        return super().init(flatten_sharded_tensors(state_dict), is_coordinator)

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        all_plans = dedup_tensors(all_plans)
        return super().create_global_plan(all_plans)

# This is ridiculous, lets just have UberSaver that a bucket of bools
class NestedDedupRenamingTensorSaver(DefaultSavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        state_dict, self.mappings = cp_nested.flatten_state_dict(state_dict)
        state_dict = flatten_sharded_tensors(state_dict)
        return super().init(state_dict, is_coordinator)

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        return dataclasses.replace(plan, planner_data=self.mappings)

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        all_plans = dedup_tensors(all_plans)
        global_plan, metadata = super().create_global_plan(all_plans)
        merged_mappings = reduce(lambda x, y: x | y, (p.planner_data for p in global_plan))
        self.metadata = dataclasses.replace(metadata, planner_data=merged_mappings)
        return global_plan, self.metadata

class UberSavePlanner(DefaultSavePlanner):
    def __init__(
        self,
        flatten_state_dict=True,
        flatten_sharded_tensors=True,
        dedup_replicated_tensors=True
    ):
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.dedup_replicated_tensors = dedup_replicated_tensors

    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        if self.flatten_state_dict:
            state_dict, self.mappings = cp_nested.flatten_state_dict(state_dict)
        if self.flatten_sharded_tensors:
            state_dict = flatten_sharded_tensors(state_dict)
        return super().init(state_dict, is_coordinator)

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        return plan

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        if self.dedup_replicated_tensors:
            all_plans = dedup_tensors(all_plans)

        global_plan, metadata = super().create_global_plan(all_plans)

        if self.flatten_state_dict:
            merged_mappings = reduce(lambda x, y: x | y, (p.planner_data for p in global_plan))
            metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

        if not validate_global_plan(global_plan, metadata):
            raise ValueError("Failed to validate global plan")

        return global_plan, metadata

# This is actually not needed if all transforms are writen in terms of traverse_state_dict
class StateDictEditor:
    def __init__(self, state_dict):
        self.original = state_dict

    def apply_mappin(self, mapping):
        pass

    def change_value(self, str_of_path, value):
        pass


class UberLoadPlanner(DefaultLoadPlanner):
    def __init__(
        self,
        flatten_state_dict=True,
        flatten_sharded_tensors=True,
    ):
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors

    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        if self.flatten_sharded_tensors:
            state_dict = flatten_sharded_tensors(state_dict)

        self.original_state_dict = state_dict        
        if self.flatten_state_dict:
            state_dict, self.mappings = cp_nested.flatten_state_dict(state_dict)
            # Note that we don't need the checkpoint mappings as we assume
            # that they are compatible with the one we just computed

        return super().init(state_dict, metadata, is_coordinator)

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        cp_nested.set_element(
            self.original_state_dict,
            self.mappings[read_item.dest_index.fqn],
            torch.load(value)
        )

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        obj = cp_nested.get_element(self.original_state_dict, self.mappings[index.fqn])
        return find_tensor_shard(obj, index)


def element_wise_sub(a, b):
    return [i_a - i_b for i_a, i_b in zip(a,b)]

class ReaderWithOffset(DefaultLoadPlanner):
    def __init__(self, fqn_to_offset) -> None:
        super().__init__()
        # str ->tuple(offset, size)
        self.fqn_to_offset = fqn_to_offset
    
    def create_local_plan(self) -> LoadPlan:
        requests = []
        self.translation = {}
        for fqn, obj in self.state_dict.items():
            md = self.metadata.state_dict_metadata[fqn]
            if not isinstance(obj, ShardedTensor):
                requests += _create_read_items(fqn, md, obj)
                continue

            if fqn not in self.fqn_to_offset:
                requests += _create_read_items(fqn, md, obj)
                continue
            
            offset = self.fqn_to_offset[fqn]

            assert len(obj.local_shards()) == 1
            original_shard = obj.local_shards()[0]
            shard_md = copy.deepcopy(original_shard.metadata)
            shard_md.shard_offsets = element_wise_add(shard_md.shard_offsets, offset)
            local_shards = [Shard(original_shard.tensor, shard_md)]

            reqs = _create_sharded_read_items(fqn, md, local_shards)
            # The WriteItems will have a displaced MetadataIndex, fix it.
            # BTW, we should change _create_sharded_read_items to have more ergnomic API
            for wi in reqs:
                original_offset = element_wise_sub(wi.dest_index.offset, offset)
                original_index = dataclasses.replace(wi.dest_index, offset=torch.Size(original_offset))
                self.translation[wi.dest_index] = original_index

            requests += reqs
        return LoadPlan(requests)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        return super().lookup_tensor(self.translation.get(index, index))
