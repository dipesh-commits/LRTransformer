import os
import torch
import numpy as np
# import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import data_prepare


# class S3DIS(Dataset):
#     def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1, inlier_data=[], inlier_label=[]):
#         super().__init__()
#         self.inlier_data = inlier_data
#         self.inlier_label = inlier_label
#         self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
#         # data_list = sorted(os.listdir(data_root))
#         # data_list = [item[:-4] for item in data_list if 'Area_' in item]
#         # if split == 'train':
#         #     self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
#         # else:
#         #     self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
#         # for item, data in enumerate(self.inlier_data):
#         #     sa_create("shm://{}".format(item), data)
#             # if not os.path.exists("/dev/shm/{}".format(item)):
#             #     data_path = os.path.join(data_root, item + '.npy')
#             #     data = np.load(data_path)  # xyzrgbl, N*7
#             #     sa_create("shm://{}".format(item), data)
#         self.data_idx = np.arange(len(self.inlier_data))
#         print("Totally {} samples in {} set.".format(len(self.data_idx), split))

#     def __getitem__(self, idx):
#         # data_idx = self.data_idx[idx % len(self.data_idx)]
#         # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
#         coord, feat, label = self.inlier_data[idx][0:3], self.inlier_data[idx][3:], self.inlier_label[idx][:]
#         # coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
#         coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
#         return coord, feat, label

#     def __len__(self):
#         return len(self.data_idx) * self.loop

class S3DIS(Dataset):
	def __init__(self, inlier_data, inlier_label, inlier_co, neighbor_data, neighbor_label, neighbor_co, transform = None, target_transform = None):
		self.inlier_data = inlier_data
		self.inlier_label = inlier_label
		self.inlier_co = inlier_co
		self.neighbor_data = neighbor_data
		self.neighbor_label = neighbor_label
		self.neighbor_co = neighbor_co
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.inlier_data)
	
	def __getitem__(self, idx):
		inlier_points = self.inlier_data[idx]
		inlier_labels = self.inlier_label[idx]
		inlier_counts = self.inlier_co[idx]

		neighbor_points = self.neighbor_data[idx]
		neighbor_labels = self.neighbor_label[idx]
		neighbor_counts = self.neighbor_co[idx]

		if self.transform:
			point = self.transform(point)
		if self.target_transform:
			label = self.target_transform(label)
		return torch.tensor(inlier_points), torch.tensor(inlier_labels), torch.tensor(inlier_counts), torch.tensor(neighbor_points), torch.tensor(neighbor_labels), torch.tensor(neighbor_counts)