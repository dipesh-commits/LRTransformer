import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import h5py
import sys
from class_util import classes_s3dis, classes_nyu40, classes_kitti
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import math
import networkx as nx
import time
import matplotlib.pyplot as plt
from lrtransformer import *
import glob
import open3d as o3d

np.random.seed(0)
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
NUM_RESTARTS = 10
FEATURE_SIZE = 13
LITE = None
# TEST_AREAS = ['1','2','3','4','5','6']
TEST_AREAS = ['scannet']

resolution = 0.1
add_threshold = 0.5
rmv_threshold = 0.5
cluster_threshold = 8
save_results = True
cross_domain = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []
restart_scoring = 'np'
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

for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		TEST_AREAS = sys.argv[i+1].split(',')
	elif sys.argv[i]=='--save':
		save_results = True
	elif sys.argv[i]=='--cross-domain':
		cross_domain = True
	elif sys.argv[i]=='--train-area':
		TRAIN_AREA = sys.argv[i+1]
	elif sys.argv[i]=='--scoring':
		restart_scoring = sys.argv[i+1]
	elif sys.argv[i]=='--resolution':
		resolution = float(sys.argv[i+1])
	elif sys.argv[i]=='--lite':
		LITE = int(sys.argv[i+1])

for AREA in TEST_AREAS:
	# tf.compat.v1.reset_default_graph()
	if cross_domain:
		MODEL_PATH = 'models/cross_domain/lrgnet_%s.ckpt' % TRAIN_AREA
	else:
		if FEATURE_SIZE==6:
			MODEL_PATH = 'models/lrgnet_model%s_xyz.ckpt'%AREA
		elif FEATURE_SIZE==9:
			MODEL_PATH = 'models/lrgnet_model%s_xyzrgb.ckpt'%AREA
		elif FEATURE_SIZE==12:
			MODEL_PATH = 'models/lrgnet_model%s_xyzrgbn.ckpt'%AREA
		else:
			# use full set of features
			if NUM_INLIER_POINT!=512 or NUM_NEIGHBOR_POINT!=512:
				# MODEL_PATH = 'models/lrgnet_model5_%s_i_%d_j_%d.ckpt'%(AREA, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT)
				MODEL_PATH = 'models/lrgnet_model5_i_%d_j_%d.pth'%(NUM_INLIER_POINT, NUM_NEIGHBOR_POINT)
			elif LITE is not None:
				MODEL_PATH = 'models/lrgnet_model%s_lite_%d.ckpt'%(AREA, LITE)
			else:
				# MODEL_PATH = 'models/latest_s3dis/lrgnet_model105.pth'    # for ss3dis test
				# MODEL_PATH = 'models/latest_scannet/lrgnet_model.pth'    # for s3dis test
				MODEL_PATH = 'models/latest_scannet/lrgnet_model.pth'    # for scannet tesst
				# MODEL_PATH = 'models/scannet/lrgnet_model.pth'
				# MODEL_PATH = 'models/s3dis/branch_increase_more/lrgnet_model5.pth'    # using s3dis model for scannet test
	# config = tf.compat.v1.ConfigProto()
	# config.gpu_options.allow_growth = True
	# config.allow_soft_placement = True
	# config.log_device_placement = False
	# sess = tf.compat.v1.Session(config=config)
	model = LRTransformer(1, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
	model.load_state_dict(torch.load(MODEL_PATH))
	model = model.to(DEVICE)
	model.eval()
	print('Restored from %s'%MODEL_PATH)


	if AREA in ['scannet', 's3dis', 'kitti_train', 'kitti_val']:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/%s.h5' % AREA)
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_test/s3dis_area%s.h5' % AREA)
	classes = classes_kitti if 'kitti' in AREA else classes_nyu40 if AREA=='scannet' else classes_s3dis
	room_name_file = 'data/%s_room_name.txt' % AREA
	if os.path.exists(room_name_file):
		room_names = open(room_name_file, 'r').read().split('\n')
	else:
		room_names = None
	# sample_file = open('data/s3dis_sampled.txt', 'r')
	# sample_list = set(sample_file.read().split('\n'))
	
	# sample_file.close()
	logger.debug(len(all_points))
	for room_id in range(len(all_points)):
#	for room_id in [162, 157, 166, 169, 200]:
#	for room_id in [10, 44, 87, 111, 198]:
		# if room_names is not None and not '_'.join(room_names[room_id].split())+'.h5' in sample_list:
		# 	print('here')
		# 	continue
		unequalized_points = all_points[room_id]
		obj_id = all_obj_id[room_id]
		cls_id = all_cls_id[room_id]

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
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]
		xyz = points[:,:3]
		rgb = points[:,3:6]
		room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

		#compute normals
		normals = []
		curvatures = []
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
			normals.append(np.fabs(V[2]))
			curvature = S[2] / (S[0] + S[1] + S[2])
			curvatures.append(np.fabs(curvature))
		curvatures = np.array(curvatures)
		curvatures = curvatures/curvatures.max()
		normals = np.array(normals)
		if FEATURE_SIZE==6:
			points = np.hstack((xyz, room_coordinates)).astype(np.float32)
		elif FEATURE_SIZE==9:
			points = np.hstack((xyz, room_coordinates, rgb)).astype(np.float32)
		elif FEATURE_SIZE==12:
			points = np.hstack((xyz, room_coordinates, rgb, normals)).astype(np.float32)
		else:
			points = np.hstack((xyz, room_coordinates, rgb, normals, curvatures.reshape(-1,1))).astype(np.float32)
		comp_time_analysis['feature'].append(time.time() - t1)

		point_voxels = np.round(points[:,:3]/resolution).astype(int)
		cluster_label = np.zeros(len(points), dtype=int)
		cluster_id = 1
		visited = np.zeros(len(point_voxels), dtype=bool)
		inlier_points = np.zeros((1, NUM_INLIER_POINT, FEATURE_SIZE), dtype=np.float32)
		neighbor_points = np.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=np.float32)
		input_add = np.zeros((1, NUM_NEIGHBOR_POINT), dtype=np.int32)
		input_remove = np.zeros((1, NUM_INLIER_POINT), dtype=np.int32)
		# order = curvatures
		order = np.argsort(curvatures)
		#iterate over each object in the room
		with torch.no_grad():
			inlier_points = torch.from_numpy(inlier_points).to(DEVICE)
			neighbor_points = torch.from_numpy(neighbor_points).to(DEVICE)
			input_add = torch.from_numpy(input_add).to(DEVICE)
			input_remove = torch.from_numpy(input_remove).to(DEVICE)
			for seed_id in np.arange(len(points))[order]:
				if visited[seed_id]:
					continue
				seed_voxel = point_voxels[seed_id]
				target_id = obj_id[seed_id]
				target_class = classes[cls_id[np.nonzero(obj_id==target_id)[0][0]]]
				gt_mask = obj_id==target_id
				obj_voxels = point_voxels[gt_mask]
				obj_voxel_set = set([tuple(p) for p in obj_voxels])
				original_minDims = obj_voxels.min(axis=0)
				original_maxDims = obj_voxels.max(axis=0)
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

						#normal way of testing
						global cluster_id, start_time
						visited[currentMask] = True
						if np.sum(currentMask) > cluster_threshold:
							cluster_label[currentMask] = cluster_id
							cluster_id += 1
							iou = 1.0 * np.sum(np.logical_and(gt_mask,currentMask)) / np.sum(np.logical_or(gt_mask,currentMask))
							print('room %d target %3d %.4s: step %3d %4d/%4d points IOU %.3f add %.3f rmv %.3f %s'%(room_id, target_id, target_class, steps, np.sum(currentMask), np.sum(gt_mask), iou, add_acc, rmv_acc, reason))


					#determine the current points and the neighboring points
					t = time.time()
					currentPoints = points[currentMask, :].copy()
					newMinDims = minDims.copy()	
					newMaxDims = maxDims.copy()	
					newMinDims -= 1
					newMaxDims += 1
					mask = np.logical_and(np.all(point_voxels>=newMinDims,axis=1), np.all(point_voxels<=newMaxDims, axis=1))
					mask = np.logical_and(mask, np.logical_not(currentMask))
					mask = np.logical_and(mask, np.logical_not(visited))
					expandPoints = points[mask, :].copy()
					expandClass = obj_id[mask] == target_id
					rejectClass = obj_id[currentMask] != target_id
					
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
					currentPoints, center = torch.from_numpy(currentPoints).to(DEVICE), torch.from_numpy(center).to(DEVICE)
					inlier_points[0,:,:] = currentPoints[subset, :]
					inlier_points[0,:,:2] -= center[:2]
					inlier_points[0,:,6:] -= center[6:]
					input_remove[0,:] = torch.from_numpy(np.array(rejectClass)).to(DEVICE)[subset]
					if len(expandPoints) >= NUM_NEIGHBOR_POINT:
						subset = np.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
					else:
						subset = list(range(len(expandPoints))) + list(np.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
					expandPoints = torch.from_numpy(expandPoints).to(DEVICE)
					neighbor_points[0,:,:] = expandPoints[subset, :]
					input_add[0,:] = torch.from_numpy(np.array(expandClass)).to(DEVICE)[subset]
					comp_time_analysis['current_neighbor'].append(time.time() - t)
					t = time.time()

					rmv, add = model([inlier_points, neighbor_points])
					
					add_acc = torch.mean(torch.eq(torch.argmax(add,dim=-1), input_add.type(torch.long)).type(torch.float))
					rmv_acc = torch.mean(torch.eq(torch.argmax(rmv,dim=-1), input_remove.type(torch.long)).type(torch.float))
					# ls, add,add_acc, rmv,rmv_acc = sess.run([net.loss, net.add_output, net.add_acc, net.remove_output, net.remove_acc],
					# 	{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})
					comp_time_analysis['current_net'].append(time.time() - t)
					t = time.time()

					add_conf = nn.Softmax(dim=-1)(add[0])[:,1]
					rmv_conf = nn.Softmax(dim=-1)(rmv[0])[:,1]
					# add_mask = add_conf > add_threshold
					# rmv_mask = rmv_conf > rmv_threshold
					add_mask = torch.rand(len(add_conf),device=DEVICE) < add_conf 
					rmv_mask = torch.rand(len(rmv_conf),device=DEVICE) < rmv_conf
	#				add_mask = input_add[0].astype(bool)
	#				rmv_mask = input_remove[0].astype(bool)
					addPoints = neighbor_points[0,:,:][add_mask]
					addPoints[:,:2] += center[:2]
					addVoxels =torch.round(addPoints[:,:3]/resolution).cpu().detach().numpy().astype(int)
					addSet = set([tuple(p) for p in addVoxels])

					rmvPoints = inlier_points[0,:,:][rmv_mask]
					rmvPoints[:,:2] += center[:2]
					rmvVoxels = torch.round(rmvPoints[:,:3]/resolution).cpu().detach().numpy().astype(int)
					rmvSet = set([tuple(p) for p in rmvVoxels])


					updated = False
					iou = 1.0 * np.sum(np.logical_and(gt_mask,currentMask)) / np.sum(np.logical_or(gt_mask,currentMask))
	#				print('%d/%d points %d outliers %d/%d add %d/%d rmv %.2f iou'%(np.sum(np.logical_and(currentMask, gt_mask)), np.sum(gt_mask),
	#					np.sum(np.logical_and(gt_mask==0, currentMask)), len(addSet), len(expandPoints), len(rmvSet), len(currentPoints), iou))
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
		print('%s %d points: %.2fs' % (room_names[room_id] if room_names is not None else '', len(unequalized_points), time.time() - t1))

		#calculate statistics 
		gt_match = 0
		match_id = 0
		dt_match = np.zeros(cluster_label.max(), dtype=bool)
		cluster_label2 = np.zeros(len(cluster_label), dtype=int)
		room_iou = []
		unique_id, count = np.unique(obj_id, return_counts=True)
		for k in range(len(unique_id)):
			i = unique_id[np.argsort(count)][::-1][k]
			best_iou = 0
			for j in range(1, cluster_label.max()+1):
				if not dt_match[j-1]:
					iou = 1.0 * np.sum(np.logical_and(obj_id==i, cluster_label==j)) / np.sum(np.logical_or(obj_id==i, cluster_label==j))
					best_iou = max(best_iou, iou)
					if iou > 0.5:
						dt_match[j-1] = True
						gt_match += 1
						cluster_label2[cluster_label==j] = k+1
						break
			room_iou.append(best_iou)
		for j in range(1,cluster_label.max()+1):
			if not dt_match[j-1]:
				cluster_label2[cluster_label==j] = j + obj_id.max()
		prc = np.mean(dt_match)
		rcl = 1.0 * gt_match / len(set(obj_id))
		room_iou = np.mean(room_iou)

		nmi = normalized_mutual_info_score(obj_id,cluster_label)
		ami = adjusted_mutual_info_score(obj_id,cluster_label)
		ars = adjusted_rand_score(obj_id,cluster_label)
		agg_nmi.append(nmi)
		agg_ami.append(ami)
		agg_ars.append(ars)
		agg_prc.append(prc)
		agg_rcl.append(rcl)
		agg_iou.append(room_iou)
		print("Area %s room %d NMI: %.2f AMI: %.2f ARS: %.2f PRC: %.2f RCL: %.2f IOU: %.2f"%(str(AREA), room_id, nmi,ami,ars, prc, rcl, room_iou))

		comp_time_analysis['neighbor'].append(sum(comp_time_analysis['current_neighbor']))
		comp_time_analysis['iter_neighbor'].extend(comp_time_analysis['current_neighbor'])
		comp_time_analysis['current_neighbor'] = []
		comp_time_analysis['net'].append(sum(comp_time_analysis['current_net']))
		comp_time_analysis['iter_net'].extend(comp_time_analysis['current_net'])
		comp_time_analysis['current_net'] = []
		comp_time_analysis['inlier'].append(sum(comp_time_analysis['current_inlier']))
		comp_time_analysis['iter_inlier'].extend(comp_time_analysis['current_inlier'])
		comp_time_analysis['current_inlier'] = []

		#save point cloud results to file
		if save_results:
			color_sample_state = np.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(np.max(cluster_label2)+1,3))
			obj_color[0] = [100,100,100]
			unequalized_points[:,3:6] = obj_color[cluster_label2,:][unequalized_idx]
			if AREA == 'scannet':
				savePLY('results/testing/scannet%d.ply'%save_id, unequalized_points)
			else:
				savePLY('results/testing/s3dis/%d.ply'%save_id, unequalized_points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(np.mean(agg_nmi), np.std(agg_nmi),np.mean(agg_ami),np.std(agg_ami),np.mean(agg_ars),np.std(agg_ars),
	np.mean(agg_prc), np.std(agg_prc), np.mean(agg_rcl), np.std(agg_rcl), np.mean(agg_iou), np.std(agg_iou)))
total_time = 0
for i in list(comp_time_analysis.keys()):
	if not i.startswith('current'):
		comp_time_analysis['std_'+i] = np.std(comp_time_analysis[i])
		comp_time_analysis[i] = np.mean(comp_time_analysis[i])
		total_time += comp_time_analysis[i]
for i in comp_time_analysis:
	if not i.startswith('current') and not i.startswith('std'):
		print('%10s %6.2f+-%5.2fs %4.1f' % (i, comp_time_analysis[i], comp_time_analysis['std_'+i], 100.0 * comp_time_analysis[i] / total_time))
