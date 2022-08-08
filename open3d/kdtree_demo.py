import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("copy_of_fragment.pcd")
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
print(pcd.points[100])
pcd.paint_uniform_color([0.5, 0.5, 0.5])
#[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
#print(k, idx)
#np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

pcd.colors[1200] = [1, 0, 0]
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1200], 5)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

o3d.visualization.draw_geometries([pcd])
