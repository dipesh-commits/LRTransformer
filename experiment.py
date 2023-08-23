import pc_utils
import numpy as np
import open3d as o3d

file_path = 'predict_data/tabletop_items.pts'

pcd = o3d.io.read_point_cloud('predict_data/tabletop_items.pts')

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# room, _, _ = pc_utils.loadFromH5(file_path)
# points, unequalize_pts, normal_grid = pc_utils.equalize(room)

# pc_utils.compute_normals(points, debug=True)

# print('>>>>>')
# pts = pc_utils.compute_normals_eigen(points, unequalize_pts, normal_grid)

# pc_utils.visualize_pc(pts, normals=True)
