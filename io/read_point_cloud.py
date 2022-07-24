import numpy as np
import os

def read_point_cloud(filename):
    suffix = os.path.splitext(filename)[1]
    assert suffix in['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(filename, dtype=np.float32).reshape(-1, 4) #read x, y, z, intensity
    else:
        raise NotImplementedError


sample_file = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti/training/velodyne/000020.bin"
point_cloud = read_point_cloud(sample_file)
print(point_cloud.shape)