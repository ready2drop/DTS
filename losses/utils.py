import torch

import numpy as np

def eucl_distance(mask: torch.Tensor, dtype: torch.dtype):
    """
    torch implementation of the code below
    https://github.com/scipy/scipy/blob/686422c4f0a71be1b4258309590fd3e9de102e18/scipy/ndimage/_morphology.py#L2318
    """
    ft = torch.zeros((mask.ndim,) + mask.shape, dtype=dtype).to(mask.device)
    # dt = ft - torch.stack(torch.meshgrid([torch.arange(0, size) for size in mask.shape]), dim=0).to(mask.device)
    dt = ft - torch.Tensor(np.indices(mask.shape)).to(mask.device)
    dt = torch.mul(dt, dt)
    dt = torch.sum(dt, dim=0)
    dt = torch.sqrt(dt)

    return dt

def one_hot2dist(seg: torch.Tensor):
    K = seg.shape[0]
    res = torch.zeros_like(seg, dtype=seg.dtype).to(seg.device)

    for k in range(K):
        posmask = seg[k].bool()
        negmask = ~posmask
        neg_dist = eucl_distance(negmask, seg.dtype) * negmask
        pos_dist = (eucl_distance(posmask, seg.dtype) - 1) * posmask
        res[k] = neg_dist - pos_dist

    return res

def dist_map_transform():
    return one_hot2dist