"""
engine/kv_cache.py

KV cache manager. Dynamic (torch.cat) in Phase 1, pre-allocated fixed-size tensor in Phase 2, optional paged/block cache in Phase 4.  |  Phase 1 -> 2 -> 4
"""
import torch 
class KVCache():
    def __init__(self,n_layer, dtype=None,device=None):
        self.n_layer = n_layer
        self.dtype = dtype
        self.device = device
        self.keys = [None]*n_layer
        self.values = [None]*n_layer
    @property
    def length(self):
        if self.keys[0] is None:
            return 0
        else:
            return self.keys[0].shape[2]

    def update(self, layer_idx,k ,v):
        """k,v-> poora (purana+nya)"""
        k,v = self._to_storage(k),self._to_storage(v)
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx]= torch.cat([self.keys[layer_idx],k],dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], v], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

        
    def reset(self):
        self.keys = [None]*self.n_layer
        self.values = [None]*self.n_layer
        

    def _to_storage(self, t):
        if self.dtype is None and self.device is None:
            return t
        return t.to(device=self.device or t.device, dtype=self.dtype or t.dtype)




