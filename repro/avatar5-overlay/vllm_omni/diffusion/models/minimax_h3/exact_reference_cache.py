"""Opt-in bounded memoization of fixed-model reference preprocessing.

Only identical RGB images, prompt text, labels and component identities match.
All DiT ranks agree on hits before bypassing any encoder collectives. Returned
tensors are cloned so downstream code cannot mutate a stored cache value.
"""
from collections import OrderedDict
import hashlib
import json
import logging
import os

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


def content_key(namespace, metadata, images):
    if not images or any(image.mode != 'RGB' for image in images):
        return None
    header = [namespace, metadata, [(image.mode, image.size) for image in images]]
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
    for image in images:
        data = image.tobytes()
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    return namespace + ':' + digest.hexdigest()


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return value


def get(owner, *, group, rank, world_size, key_factory):
    if (os.environ.get('H3_REF2VA_EXACT_REFERENCE_CACHE') != '1'
            or not getattr(owner, '_h3_reference_cache_request', True)):
        return None, None
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {'key': key_factory()}
        except Exception as exc:
            payload[0] = {'error': type(exc).__name__}
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0, group=group, device=owner.device)
    if 'error' in payload[0]:
        raise ValueError('Reference cache key failed: ' + payload[0]['error'])
    key = payload[0]['key']
    if key is None:
        return None, None
    if not hasattr(owner, '_h3_exact_reference_cache'):
        owner._h3_exact_reference_cache = OrderedDict()
    cache = owner._h3_exact_reference_cache
    hit = key in cache
    if world_size > 1:
        agreement = torch.tensor([int(hit)], dtype=torch.int32, device=owner.device)
        dist.all_reduce(agreement, op=dist.ReduceOp.MIN, group=group)
        hit = bool(agreement.item())
    if rank == 0:
        log.info('H3_EXACT_REF_CACHE %s', json.dumps({'namespace': key.split(':')[0],
            'hit': hit, 'entries': len(cache)}))
    if not hit:
        return key, None
    cache.move_to_end(key)
    return key, clone_tree(cache[key])


def put(owner, key, value):
    if key is None:
        return
    cache = owner._h3_exact_reference_cache
    cache[key] = clone_tree(value)
    cache.move_to_end(key)
    while len(cache) > 8:
        cache.popitem(last=False)
