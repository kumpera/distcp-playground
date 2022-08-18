import math
from typing import Tuple

import torch
import torch.distributed.rpc as rpc
from torch.distributed._shard._utils import narrow_tensor_by_index


class SpacesControler:
    def __init__(self):
        self.client_map = {}
        self.world = {}

    def _register(self, id, rref):
        # print(f"[controler] client:{id} registered {rref}")
        self.client_map[id] = rref

    def add_shard(self, name, id, global_size, offset, length):
        _, mode, shard_list = self.world.setdefault(name, (global_size, "sharded", []))
        if mode != "sharded":
            raise ValueError(f"can't mix {mode} with sharded")
        shard_list.append((self.client_map[id], offset, length))
    
    def add_chunk(self, name, id, global_size, offset, length):
        _, mode, chunk_list = self.world.setdefault(name, (global_size, "linear", []))
        if mode != "linear":
            raise ValueError(f"can't mix {mode} with linear")
        chunk_list.append((self.client_map[id], offset, length))

    def get_tensor_info(self, name) -> Tuple:
        return self.world[name]

class RemoteTensor:
    def __init__(self, space, name):
        self.space = space
        self.name = name
        self.size = self.mode = self.chunks = None
    
    def _lazy_init(self):
        if self.size is not None:
            return
        self.size, self.mode, self.chunks = self.space.get_tensor_info(self.name)

    def is_local(self):
        self._lazy_init()
        if len(self.chunks) == 1:
            client, _, _ = self.chunks[0]
            return client.is_owner()
        return False

    def get_localfyness(self):
        """"Return the fraction of this tensor that's local"""
        self._lazy_init()
        numel = 0
        for client, _, length in self.chunks:
            if client.is_owner():
                if isinstance(length, int):
                    numel += length
                else:
                    numel += math.prod(length)
        return float(numel) / math.prod(self.size)

    def _get_shard(self, client, shard_id):
        if client.is_owner():
            return Spaces._local_instance.get_shard(self.name, shard_id)
        return client.rpc_sync().get_shard(self.name, shard_id)

    def _get_chunk(self, client, chunk_id):
        if client.is_owner():
            return Spaces._local_instance.get_chunk(self.name, chunk_id)
        return client.rpc_sync().get_chunk(self.name, chunk_id)

    def local_copy(self):
        self._lazy_init()
        if self.mode == "sharded":
            if len(self.chunks) == 1:
                client, _, _ = self.chunks[0]
                shard_id = 0 #TODO
                return self._get_shard(client, shard_id)
            res = torch.empty(self.size) #XXX dtype, device, etc, etc, etc
            for client, offset, length in self.chunks:
                shard_id = 0 #TODO
                shard = self._get_shard(client, shard_id)
                narrow_tensor_by_index(res, offset, length).copy_(shard)
            return res
        elif self.mode == "linear":
            if len(self.chunks) == 1:
                client, _, _ = self.chunks[0]
                chunk_id = 0 #TODO
                return self._get_chunk(client, chunk_id).view(self.size)
            res = torch.empty(self.size) #XXX dtype, device, etc, etc, etc
            res_flat = torch.flatten(res)
            for client, offset, length in self.chunks:
                chunk_id = 0 #TODO
                chunk = self._get_chunk(client, chunk_id)
                torch.narrow(res_flat, 0, offset, length).copy_(chunk)
            return res
        else:
            raise ValueError(f"can't handle {self.mode} yet")

class Spaces:
    _global_controler = None
    _local_instance = None
    # TODO Make this cstor actually init a workspace/namespace
    def __init__(self):
        if Spaces._local_instance is not None:
            raise ValueError("Already registered instance of Space")
        self.id = rpc.get_worker_info().id
        self._controler = rpc.remote("worker0", Spaces._init_central_space)
        self._controler.rpc_sync()._register(self.id, rpc.RRef(self))
        self._local_shards = {}
        self._local_chunks = {}
        Spaces._local_instance = self

    @staticmethod
    def _init_central_space():
        if Spaces._global_controler is None:
            Spaces._global_controler = SpacesControler()
        return Spaces._global_controler


    def register(self, name, local_tensor):
        self.register_shard(
            name,
            local_tensor.size(),
            [0] * len(local_tensor.size()),
            local_tensor.size(),
            local_tensor
        )

    def register_shard(self, name, global_size, offset, length, local_tensor):
        self._local_shards[name] = (global_size, offset, length, local_tensor)
        self._controler.rpc_sync().add_shard(name, self.id, global_size, offset, length)


    def register_linear(self, name, global_size, offset, length, local_tensor):
        self._local_chunks[name] = (global_size, offset, length, local_tensor)
        self._controler.rpc_sync().add_chunk(name, self.id, global_size, offset, length)

    def get_shard(self, name, shard_id):
        return self._local_shards[name][3]

    def get_chunk(self, name, chunk_id):
        return self._local_chunks[name][3]

    def get_tensor(self, name):
        # TODO check cache and resolve locally if possible
        return RemoteTensor(self, name)

    def get_tensor_info(self, name):
        # TODO caching (and check against local info)
        return self._controler.rpc_sync().get_tensor_info(name)
class SpacesControler:
    def __init__(self):
        self.client_map = {}
        self.world = {}

    def _register(self, id, rref):
        # print(f"[controler] client:{id} registered {rref}")
        self.client_map[id] = rref

    def add_shard(self, name, id, global_size, offset, length):
        _, mode, shard_list = self.world.setdefault(name, (global_size, "sharded", []))
        if mode != "sharded":
            raise ValueError(f"can't mix {mode} with sharded")
        shard_list.append((self.client_map[id], offset, length))
    
    def add_chunk(self, name, id, global_size, offset, length):
        _, mode, chunk_list = self.world.setdefault(name, (global_size, "linear", []))
        if mode != "linear":
            raise ValueError(f"can't mix {mode} with linear")
        chunk_list.append((self.client_map[id], offset, length))

    def get_tensor_info(self, name) -> Tuple:
        return self.world[name]

class RemoteTensor:
    def __init__(self, space, name):
        self.space = space
        self.name = name
        self.size = self.mode = self.chunks = None
    
    def _lazy_init(self):
        if self.size is not None:
            return
        self.size, self.mode, self.chunks = self.space.get_tensor_info(self.name)

    def is_local(self):
        self._lazy_init()
        if len(self.chunks) == 1:
            client, _, _ = self.chunks[0]
            return client.is_owner()
        return False

    def get_localfyness(self):
        """"Return the fraction of this tensor that's local"""
        self._lazy_init()
        numel = 0
        for client, _, length in self.chunks:
            if client.is_owner():
                if isinstance(length, int):
                    numel += length
                else:
                    numel += math.prod(length)
        return float(numel) / math.prod(self.size)

    def _get_shard(self, client, shard_id):
        if client.is_owner():
            return Spaces._local_instance.get_shard(self.name, shard_id)
        return client.rpc_sync().get_shard(self.name, shard_id)

    def _get_chunk(self, client, chunk_id):
        if client.is_owner():
            return Spaces._local_instance.get_chunk(self.name, chunk_id)
        return client.rpc_sync().get_chunk(self.name, chunk_id)

    def local_copy(self):
        self._lazy_init()
        if self.mode == "sharded":
            if len(self.chunks) == 1:
                client, _, _ = self.chunks[0]
                shard_id = 0 #TODO
                return self._get_shard(client, shard_id)
            res = torch.empty(self.size) #XXX dtype, device, etc, etc, etc
            for client, offset, length in self.chunks:
                shard_id = 0 #TODO
                shard = self._get_shard(client, shard_id)
                narrow_tensor_by_index(res, offset, length).copy_(shard)
            return res
        elif self.mode == "linear":
            if len(self.chunks) == 1:
                client, _, _ = self.chunks[0]
                chunk_id = 0 #TODO
                return self._get_chunk(client, chunk_id).view(self.size)
            res = torch.empty(self.size) #XXX dtype, device, etc, etc, etc
            res_flat = torch.flatten(res)
            for client, offset, length in self.chunks:
                chunk_id = 0 #TODO
                chunk = self._get_chunk(client, chunk_id)
                torch.narrow(res_flat, 0, offset, length).copy_(chunk)
            return res
        else:
            raise ValueError(f"can't handle {self.mode} yet")

class Spaces:
    _global_controler = None
    _local_instance = None
    # TODO Make this cstor actually init a workspace/namespace
    def __init__(self):
        if Spaces._local_instance is not None:
            raise ValueError("Already registered instance of Space")
        self.id = rpc.get_worker_info().id
        self._controler = rpc.remote("worker0", Spaces._init_central_space)
        self._controler.rpc_sync()._register(self.id, rpc.RRef(self))
        self._local_shards = {}
        self._local_chunks = {}
        Spaces._local_instance = self

    @staticmethod
    def _init_central_space():
        if Spaces._global_controler is None:
            Spaces._global_controler = SpacesControler()
        return Spaces._global_controler


    def register(self, name, local_tensor):
        self.register_shard(
            name,
            local_tensor.size(),
            [0] * len(local_tensor.size()),
            local_tensor.size(),
            local_tensor
        )

    def register_shard(self, name, global_size, offset, length, local_tensor):
        self._local_shards[name] = (global_size, offset, length, local_tensor)
        self._controler.rpc_sync().add_shard(name, self.id, global_size, offset, length)


    def register_linear(self, name, global_size, offset, length, local_tensor):
        self._local_chunks[name] = (global_size, offset, length, local_tensor)
        self._controler.rpc_sync().add_chunk(name, self.id, global_size, offset, length)

    def get_shard(self, name, shard_id):
        return self._local_shards[name][3]

    def get_chunk(self, name, chunk_id):
        return self._local_chunks[name][3]

    def get_tensor(self, name):
        # TODO check cache and resolve locally if possible
        return RemoteTensor(self, name)

    def get_tensor_info(self, name):
        # TODO caching (and check against local info)
        return self._controler.rpc_sync().get_tensor_info(name)
