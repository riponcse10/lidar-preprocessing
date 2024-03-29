# Lidar Preprocessing
This repository contains common utility functions to preprocess LiDAR point clouds. Such preprocessing steps are required to perform different operations, such as object detection, downsampling, voxelization, and so on.

Currently implemented preprocessing functions:
1. Load point cloud data from .bin file
2. Calculate Chamfer distance
3. Create birds eye view image from point cloud
4. Create range image from point cloud
5. DBScan
6. KdTree
7. Plane Segmentation
8. Voxelize
9. Compression with dracopy


Used the following libraries:
1. Open3d
2. pytorch3d
3. torch-points3d
4. DracoPy
