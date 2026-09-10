"""
engine/kv_cache.py

KV cache manager. Dynamic (torch.cat) in Phase 1, pre-allocated fixed-size tensor in Phase 2, optional paged/block cache in Phase 4.  |  Phase 1 -> 2 -> 4
"""

