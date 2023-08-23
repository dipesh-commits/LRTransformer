import h5py
import numpy as np
import glob
import os
import open3d as o3d

from loguru import logger


def main():
    ROOT_DIR = '../data/shapenet_part_seg/hdf5_data'
    h5_files = glob.glob(ROOT_DIR+'/*.h5')
    obj_id = 0
    initial = True
    for idx, files in enumerate(h5_files):
        file_name = f'../data/shapenet_part_seg/region_grow_data/shapenet_train_{idx}.h5'
        f = h5py.File(files)
        print(f.keys())
        objects = f['data']
        object_label = f['label']
        part_label = f['pid']
        all_points = np.zeros((objects.shape[0], objects.shape[1], 5))
        count_objects = []
        for i, obj in enumerate(objects):
            count_objects.append(obj.shape[0])
            new_obj = np.hstack((obj, part_label[i].reshape(-1,1)))
            new_obj = new_obj[np.argsort(new_obj[:,3])]
            obj_id_col = np.zeros((new_obj.shape[0],1))
            new_obj = np.hstack((new_obj, obj_id_col))
            
            for j, point in enumerate(new_obj):
                if j==0:
                    new_obj_id = obj_id + 1 if initial else new_obj_id+1
                    new_obj[j][4] = new_obj_id
                    previous = new_obj[j][3]
                    initial = False
                else:
                    if previous == new_obj[j][3]:
                        new_obj[j][4] = new_obj_id
                    else:
                        new_obj_id = new_obj_id+1
                        new_obj[j][4] = new_obj_id
                        previous = new_obj[j][3]
            
            all_points[i,:,:] = new_obj
        all_points = all_points.reshape(-1,5)
        all_points[:,[3,4]] = all_points[:,[4,3]]
        count_objects = np.asarray(count_objects)
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('points',data=all_points)
            f.create_dataset('count_room', data=count_objects)  # here room means objects
        print(f"{file_name} saved sucessfully")


def visualize_h5(data_path):
    data = h5py.File(data_path, 'r')
    print(data.keys())
    points = data['points']
    count = data['count']
    
    previous_count = 0
    for i, p in enumerate(points):
        
        test_point = points[previous_count:previous_count+count[i]]
        previous_count+= count[i]
        if i==4:
            break

    normals = test_point[:,9:12]
    pcd = o3d.geometry.PointCloud()
    
    test_point = points
    # test_point = np.asarray(test_point)
    # logger.error(normals)
    pcd.points = o3d.utility.Vector3dVector(test_point[:,3:6])
    pcd.colors = o3d.utility.Vector3dVector(test_point[:,6:9])
    pcd.normals = o3d.utility.Vector3dVector(test_point[:,9:12])
    o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    data_path = '../data/staged/staged_area1.h5'
    visualize_h5(data_path)