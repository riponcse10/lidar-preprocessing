import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("copy_of_fragment.pcd")
#o3d.visualization.draw_geometries([pcd])
N = 2000

pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),center=pcd.get_center())

pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
o3d.visualization.draw_geometries([pcd])

print('voxelization')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.05)

o3d.visualization.draw_geometries([voxel_grid])