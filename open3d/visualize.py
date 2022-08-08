from open3d import *
import numpy as np
import struct

size_float = 4
list_pcd = []
with open("/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti/training/velodyne/000000.bin", "rb") as f:
    byte = f.read(size_float * 4)
    while byte:
        x, y, z, intensity = struct.unpack("ffff", byte)
        list_pcd.append([x, y, z])
        byte = f.read(size_float * 4)
np_pcd = np.asarray(list_pcd)
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(np_pcd)

open3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

data = open3d.io.read_point_cloud(filename="copy_of_fragment.pcd")
print(data.points)
open3d.visualization.draw_geometries_with_editing([data])

#open3d.visualization.read_selection_polygon_volume("copy_of_fragment.pcd")

#lets downsample the pcd

downsamplded = data.voxel_down_sample(voxel_size=0.3)
arr = np.asarray(downsamplded.points)
print(arr.shape)
print(np.max(arr.T[0]))
print(np.min(arr.T[0]))

print(np.max(arr.T[1]))
print(np.min(arr.T[1]))

print(np.max(arr.T[2]))
print(np.min(arr.T[2]))

deleted = np.delete(arr, np.where(arr < -10)[0], axis=0)
print(deleted.shape)
print(np.max(deleted.T[0]))
print(np.min(deleted.T[0]))

print(np.max(deleted.T[1]))
print(np.min(deleted.T[1]))

print(np.max(deleted.T[2]))
print(np.min(deleted.T[2]))

deleted_x = np.delete(deleted, np.where(deleted > 10)[0], axis=0)
print(deleted_x.shape)
print(np.max(deleted_x.T[0]))
print(np.min(deleted_x.T[0]))

print(np.max(deleted_x.T[1]))
print(np.min(deleted_x.T[1]))

print(np.max(deleted_x.T[2]))
print(np.min(deleted_x.T[2]))


point_cloud = open3d.geometry.PointCloud()
point_cloud.points = open3d.utility.Vector3dVector(deleted_x)

open3d.visualization.draw_geometries([point_cloud])