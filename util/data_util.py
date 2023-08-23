import numpy as np
import random
# import SharedArray as SA

import torch

from util.voxelize import voxelize
from loguru import logger


# def sa_create(name, var):
#     x = SA.create(name, var.shape, dtype=var.dtype)
#     x[...] = var[...]
#     x.flags.writeable = False
#     return x


def collate_fn(batch):
    inlier_feat, inlier_label, inlier_count, neighbor_feat, neighbor_label, neighbor_count = list(zip(*batch))
    inlier_offset, count = [], 0
    for item in inlier_feat:
        count += item.shape[0]
        inlier_offset.append(count)
    neighbor_offset, count = [], 0
    for item in neighbor_feat:
        count+= item.shape[0]
        neighbor_offset.append(count)
    # inlier_feat,inlier_label = [inlier_feat], [inlier_label]
    return inlier_feat, inlier_label, torch.IntTensor(inlier_count), torch.IntTensor(inlier_offset), neighbor_feat, neighbor_label, torch.IntTensor(neighbor_count), torch.IntTensor(neighbor_offset)


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label