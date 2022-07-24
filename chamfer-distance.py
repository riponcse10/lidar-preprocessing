#Reference : https://pytorch3d.readthedocs.io/en/latest/modules/loss.html

from pytorch3d.loss import chamfer_distance
from pytorch3d.io import IO
from io.read_point_cloud import read_point_cloud
import torch
import struct
import numpy as np

sample_file_1 = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti/training/velodyne/000020.bin"
sample_file_2 = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti/training/velodyne/000021.bin"
device = "cuda" if torch.cuda.is_available() else "cpu"

#pytorch3d library requires a ply file as the input of point cloud,
# so we write the point cloud as ply file and read them again to calculate chamfer distance
def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()



x = read_point_cloud(sample_file_1)
y = read_point_cloud(sample_file_2)
write_pointcloud("1.ply", x[:, :3])
write_pointcloud("2.ply", y[:, :3])
point_cloud1 = IO().load_pointcloud("1.ply", device)
point_cloud2 = IO().load_pointcloud("2.ply", device)
print(chamfer_distance(point_cloud1, point_cloud2))

