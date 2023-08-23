from cmath import nan
import numpy as np
import h5py
import os

import sys

import itertools
import random
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import math
# import networkx as nx
from scipy.cluster.vq import vq, kmeans
import time
# import matplotlib.pyplot as plt


import torch.nn as nn


from lrtransformer import *

import open3d as o3d


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
#   Curvatures may contain nan values.  Using 12 features instead of 13.  Hopefully does not damage results too much.
FEATURE_SIZE = 13
LITE = None
TEST_AREAS = ['1','2','3','4','5','6','scannet']
resolution = 0.001
add_threshold = 0.5
rmv_threshold = 0.5
#   Original cluster_threshold is 10.
cluster_threshold = 10
save_results = False
cross_domain = False
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []
comp_time_analysis = {
	'feature': [],
	'net': [],
	'neighbor': [],
	'inlier': [],
	'current_net' : [],
	'current_neighbor' : [],
	'current_inlier' : [],
	'iter_net' : [],
	'iter_neighbor' : [],
	'iter_inlier' : [],
}

SCAN_NAME = None
RESULTS_DIR = None

	# elif sys.argv[i]=='--save_results':
	# 	save_results = True
for i in range(len(sys.argv)):
    if sys.argv[i]=='--scan_filename':
        SCAN_NAME = str(sys.argv[i+1])
    elif sys.argv[i]=='--results_dir':
        RESULTS_DIR = str(sys.argv[i+1])
    elif sys.argv[i]=='--resolution':
        resolution = float(sys.argv[i+1])
    elif sys.argv[i]=='--lite':
        LITE = int(sys.argv[i+1])


model = LRTransformer(1,1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT,13)

MODEL_PATH = 'models/lrgnet_model105.pth'

model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()
print('Restored from %s'%MODEL_PATH)

source_cloud = o3d.io.read_point_cloud(SCAN_NAME)
if SCAN_NAME[-7:] == ".xyzrgb":
    source_cloud.colors = o3d.utility.Vector3dVector(np.asarray(source_cloud.colors) / 255)

print(np.asarray(source_cloud.points).shape)

source_cloud_np = np.asarray(np.concatenate((np.asarray(source_cloud.points), np.asarray(source_cloud.colors)), axis=1))
source_colors = np.asarray(source_cloud.colors)

down_sample = 0.005
idx_list = None
# cloud, _, idx_list = source_cloud.voxel_down_sample_and_trace(down_sample, source_cloud.get_min_bound(), source_cloud.get_max_bound())
cloud = source_cloud

cloud_center = cloud.get_center()
scale = 1/6
cloud = cloud.scale(scale, cloud_center)
cloud = cloud.translate(-1*cloud_center, relative=True)

print(f"Cloud center (post-translation):  {cloud.get_center()}")
print(f"Cloud min bound:  {cloud.get_min_bound()}")
print(f"Cloud max bound:  {cloud.get_max_bound()}")

cloud = np.asarray(np.concatenate((np.asarray(cloud.points), np.asarray(cloud.colors)), axis=1))
print(cloud.shape)

unequalized_points = cloud

#equalize resolution
t1 = time.time()
equalized_idx = []
unequalized_idx = []
equalized_map = {}
normal_grid = {}
for i in range(len(unequalized_points)):
    k = tuple(np.round(unequalized_points[i,:3]/resolution).astype(int))
    if not k in equalized_map:
        equalized_map[k] = len(equalized_idx)
        equalized_idx.append(i)
    unequalized_idx.append(equalized_map[k])
    if not k in normal_grid:
        normal_grid[k] = []
    normal_grid[k].append(i)
points = unequalized_points[equalized_idx]
logger.warning(points.shape)
# obj_id = obj_id[equalized_idx]
# cls_id = cls_id[equalized_idx]
xyz = points[:,:3]
rgb = points[:,3:6]
room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

#compute normals
normals = []
curvatures = []
# deletes = []
for i in range(len(points)):
    k = tuple(np.round(points[i,:3]/resolution).astype(int))
    neighbors = []
    for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
        kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
        if kk in normal_grid:
            neighbors.extend(normal_grid[kk])
    accA = np.zeros((3,3))
    accB = np.zeros(3)
    for n in neighbors:
        p = unequalized_points[n,:3]
        accA += np.outer(p,p)
        accB += p
    cov = accA / len(neighbors) - np.outer(accB, accB) / len(neighbors)**2
    U,S,V = np.linalg.svd(cov)
    curvature = S[2] / (S[0] + S[1] + S[2])

    #   If curvature is nan, delete this point from dataset.
    # if np.isnan(curvature):
    #     deletes.append(i)
    #     continue

    normals.append(np.fabs(V[2]))
    curvatures.append(np.fabs(curvature))


curvatures = np.array(curvatures)
curvatures = curvatures/np.nanmax(curvatures)
normals = np.array(normals)
if FEATURE_SIZE==6:
    points = np.hstack((xyz, room_coordinates)).astype(np.float32)
elif FEATURE_SIZE==9:
    points = np.hstack((xyz, room_coordinates, rgb)).astype(np.float32)
elif FEATURE_SIZE==12:
    points = np.hstack((xyz, room_coordinates, rgb, normals)).astype(np.float32)
else:
    points = np.hstack((xyz, room_coordinates, rgb, normals, curvatures.reshape(-1,1))).astype(np.float32)

#logger.error(np.round(points[:,:3]/resolution).astype(int))
logger.error(points.shape)

comp_time_analysis['feature'].append(time.time() - t1)

point_voxels = np.round(points[:,:3]/resolution).astype(int)
cluster_label = np.zeros(len(points), dtype=int)
cluster_id = 1
visited = np.zeros(len(point_voxels), dtype=bool)


inlier_points = np.zeros((1, NUM_INLIER_POINT, FEATURE_SIZE), dtype=np.float32)
neighbor_points = np.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=np.float32)
# input_add = np.zeros((1, NUM_NEIGHBOR_POINT), dtype=np.int32)
# input_remove = np.zeros((1, NUM_INLIER_POINT), dtype=np.int32)
order = np.argsort(curvatures)


with torch.no_grad():
    inlier_points = torch.from_numpy(inlier_points)
    neighbor_points = torch.from_numpy(neighbor_points)
    for seed_id in np.arange(len(points))[order]:
        if visited[seed_id]:
            continue
        seed_voxel = point_voxels[seed_id]
        currentMask = np.zeros(len(points), dtype=bool)
        
        currentMask[seed_id] = True
        minDims = seed_voxel.copy()
        maxDims = seed_voxel.copy()
        seqMinDims = minDims
        seqMaxDims = maxDims
        steps = 0
        stuck = 0
        maskLogProb = 0

        #perform region growing
        while True:

            def stop_growing(reason):
                # print(reason)
                global cluster_id, start_time
                visited[currentMask] = True
                if np.sum(currentMask) > cluster_threshold:
                    cluster_label[currentMask] = cluster_id
                    cluster_id += 1

            #determine the current points and the neighboring points
            t = time.time()
            
            currentPoints = points[currentMask, :].copy()

            newMinDims = minDims.copy()	
            newMaxDims = maxDims.copy()	
            newMinDims -= 1
            newMaxDims += 1

            # print(point_voxels)
            # print(seed_id)

            mask = np.logical_and(np.all(point_voxels>=newMinDims,axis=1), np.all(point_voxels<=newMaxDims, axis=1))
            
            mask = np.logical_and(mask, np.logical_not(currentMask))
            
            mask = np.logical_and(mask, np.logical_not(visited))
            expandPoints = points[mask, :].copy()
            
           
            
            if len(expandPoints)==0: #no neighbors (early termination)
                stop_growing('noneighbor')
                break

            if len(currentPoints) >= NUM_INLIER_POINT:
                subset = np.random.choice(len(currentPoints), NUM_INLIER_POINT, replace=False)
            else:
                subset = list(range(len(currentPoints))) + list(np.random.choice(len(currentPoints), NUM_INLIER_POINT-len(currentPoints), replace=True))
            center = np.median(currentPoints, axis=0)
            expandPoints = np.array(expandPoints)
            expandPoints[:,:2] -= center[:2]
            expandPoints[:,6:] -= center[6:]

            currentPoints = torch.from_numpy(currentPoints).to(DEVICE)
            center = torch.from_numpy(center).to(DEVICE)
            inlier_points = inlier_points.to(DEVICE)
            
            inlier_points[0,:,:] = currentPoints[subset, :]
            inlier_points[0,:,:2] -= center[:2]
            inlier_points[0,:,6:] -= center[6:]
            if len(expandPoints) >= NUM_NEIGHBOR_POINT:
                subset = np.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
            else:
                subset = list(range(len(expandPoints))) + list(np.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
            expandPoints = torch.from_numpy(expandPoints).to(DEVICE)
            neighbor_points = neighbor_points.to(DEVICE)
            neighbor_points[0,:,:] = expandPoints[subset, :]
            comp_time_analysis['current_neighbor'].append(time.time() - t)
            t = time.time()            


            rmv, add = model([inlier_points, neighbor_points])
           
            # ls, add, add_acc, rmv,rmv_acc = sess.run([net.loss, net.add_output, net.add_acc, net.remove_output, net.remove_acc],
            #     {net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})

            comp_time_analysis['current_net'].append(time.time() - t)
            t = time.time()

            # rmv, add = rmv.cpu(), add.cpu()
            
            # add_conf = scipy.special.softmax(add[0], axis=-1)[:,1]
            # rmv_conf = scipy.special.softmax(rmv[0], axis=-1)[:,1]
            
            add_conf = nn.Softmax(dim=-1)(add[0])[:,1]
            rmv_conf = nn.Softmax(dim=-1)(rmv[0])[:,1]
            # exit()


            add_mask = torch.rand(len(add_conf),device=DEVICE) < add_conf           
            rmv_mask = torch.rand(len(rmv_conf),device=DEVICE) < rmv_conf

            addPoints = neighbor_points[0,:,:][add_mask]
            addPoints[:,:2] += center[:2]            
            addVoxels = torch.round(addPoints[:,:3]/resolution)
            addVoxels = addVoxels.cpu().detach().numpy().astype(int)
            addSet = set([tuple(p) for p in addVoxels])
            
            rmvPoints = inlier_points[0,:,:][rmv_mask]
            rmvPoints[:,:2] += center[:2]
            rmvVoxels = torch.round(rmvPoints[:,:3]/resolution)
            rmvVoxels = rmvVoxels.cpu().detach().numpy().astype(int)
            rmvSet = set([tuple(p) for p in rmvVoxels])
            
            
            updated = False

            
            #this loop is not running
            for i in range(len(point_voxels)):
                if not currentMask[i] and tuple(point_voxels[i]) in addSet:
                    currentMask[i] = True
                    updated = True
                    
                if tuple(point_voxels[i]) in rmvSet:
                    currentMask[i] = False
            
            steps += 1
            comp_time_analysis['current_inlier'].append(time.time() - t)

            if updated: #continue growing
                minDims = point_voxels[currentMask, :].min(axis=0)
                maxDims = point_voxels[currentMask, :].max(axis=0)
                if not np.any(minDims<seqMinDims) and not np.any(maxDims>seqMaxDims):
                    if stuck >= 1:
                        stop_growing('stuck')
                        break
                    else:
                        stuck += 1
                else:
                    stuck = 0
                seqMinDims = np.minimum(seqMinDims, minDims)
                seqMaxDims = np.maximum(seqMaxDims, maxDims)
            else: #no matching neighbors (early termination)
                stop_growing('noexpand')
                break


#fill in points with no labels
nonzero_idx = np.nonzero(cluster_label)[0]
nonzero_points = points[nonzero_idx, :]
filled_cluster_label = cluster_label.copy()
for i in np.nonzero(cluster_label==0)[0]:
    d = np.sum((nonzero_points - points[i])**2, axis=1)
    closest_idx = np.argmin(d)
    filled_cluster_label[i] = cluster_label[nonzero_idx[closest_idx]]
cluster_label = filled_cluster_label

#save point cloud results to file
color_sample_state = np.random.RandomState(0)
obj_color = color_sample_state.randint(0,255,(np.max(cluster_label)+1,3))
obj_color[0] = [100,100,100]
unequalized_points[:,3:6] = obj_color[cluster_label,:][unequalized_idx]

# if save_results:
#     savePLY(f'data/results/lrg/{SCAN_NAME}.ply', unequalized_points)

#   Propagate segmentation results (colors of output cloud) to source cloud.
result_points = o3d.utility.Vector3dVector(unequalized_points[:, :3])
result_colors = o3d.utility.Vector3dVector(unequalized_points[:, 3:] / 255)
result_cloud = o3d.geometry.PointCloud(result_points)
result_cloud.colors = result_colors

result_cloud = result_cloud.translate(cloud_center, relative=True)
# result_cloud = result_cloud.scale(1/scale, result_cloud.get_center())

result_points = np.asarray(np.concatenate((np.asarray(result_cloud.points), np.asarray(result_cloud.colors)), axis=1))

unique_colors = np.unique(result_points[:, 3:6], axis=0)
print(f"Number of objects detected: {unique_colors.shape[0]}")

# o3d.visualization.draw_geometries([result_cloud])

# o3d.io.write_point_cloud(os.path.join(RESULTS_DIR, f"{os.path.basename(SCAN_NAME)[:-4]}_full_results_{down_sample}.pts"), result_cloud)

print("Begin recoloring.")

if idx_list is not None:
    for i in range(len(idx_list)):
        idx = idx_list[i]
        color = result_points[i, 3:]

        source_cloud_np[idx, 3:] = color

    recolored_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_cloud_np[:, 0:3]))
    recolored_cloud.colors = o3d.utility.Vector3dVector(source_cloud_np[:, 3:6])
    recolored_points = np.asarray(np.concatenate((np.asarray(recolored_cloud.points), np.asarray(recolored_cloud.colors)), axis=1))

    o3d.visualization.draw_geometries([recolored_cloud])

    save = input("Enter S to save: ")

    if save == 'S' or save == 's':
        o3d.io.write_point_cloud(os.path.join(RESULTS_DIR, f"{os.path.basename(SCAN_NAME)[:-4]}_full_results_{down_sample}.pts"), recolored_cloud)
else:
    o3d.visualization.draw_geometries([result_cloud])

    save = input("Enter S to save: ")

    if save == 'S' or save == 's':
        o3d.io.write_point_cloud(os.path.join(RESULTS_DIR, f"{os.path.basename(SCAN_NAME)[:-4]}_full_results_{down_sample}.pts"), result_cloud)

exit()
