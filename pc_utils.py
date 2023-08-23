import os
import h5py
import itertools
import numpy as np
import open3d as o3d


def visualize_pc(pts, normals=False):
	pcd = o3d.geometry.PointCloud()
	if not isinstance(pts, o3d.utility.Vector3dVector):
		pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
	# assert isinstance(pcd, o3d.utility.Vector3dVector)
	pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
	# pcd.normals = o3d.utility.Vector3dVector(pts[:,6:9])
	if normals:
		pcd.estimate_normals(
    	search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
		o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
		o3d.io.write_point_cloud('results/normals/lrg.pts',pcd)
	else:
		o3d.visualization.draw_geometries([pcd])
	


def loadFromH5(filename, load_labels=True):
	f = h5py.File(filename,'r')
	all_points = f['points'][:]
	count_room = f['count_room'][:]
	tmp_points = []
	idp = 0
	for i in range(len(count_room)):
		tmp_points.append(all_points[idp:idp+count_room[i], :])
		idp += count_room[i]
	f.close()
	room = []
	labels = []
	class_labels = []
	if load_labels:
		for i in range(len(tmp_points)):
			room.append(tmp_points[i][:,:-2])
			labels.append(tmp_points[i][:,-2].astype(int))
			class_labels.append(tmp_points[i][:,-1].astype(int))
		return room, labels, class_labels
	else:
		return tmp_points
	

def equalize(points):
	resolution = 0.1
	for room_id in range(len(points)):		
		unequalized_points = points[room_id]
		#equalize resolution
		equalized_idx = []
		equalized_set = set()
		normal_grid = {}
		for i in range(len(unequalized_points)):
			k = tuple(np.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_set:
				equalized_set.add(k)
				equalized_idx.append(i)
			if not k in normal_grid:
				normal_grid[k] = []
			normal_grid[k].append(i)
		# points -> XYZ + RGB
		points = unequalized_points[equalized_idx]
		return points, unequalized_points, normal_grid
	

def compute_normals(pts, debug=False):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
	pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
	pcd.estimate_normals()
	print(np.asarray(pcd.normals))
	# print(np.asarray(pcd.points))
	if debug:
		o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
	o3d.io.write_point_cloud('results/normals/open3d.pts',pcd)

		
def compute_normals_eigen(points, unequalized_points, normal_grid):
	resolution = 0.1
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
	normals = np.array(normals)
	curvatures = np.array(curvatures)
	curvatures = curvatures/curvatures.max()
	# print(np.hstack((points, normals)))
	return np.hstack((points, normals))