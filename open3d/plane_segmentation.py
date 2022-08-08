import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def get_color_map(pcd):
    #pcd = o3d.io.read_point_cloud(filename="copy_of_fragment.pcd")

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    o3d.visualization.draw_geometries([outlier_cloud])


pcd = o3d.io.read_point_cloud(filename="copy_of_fragment.pcd")
get_color_map(pcd)

#ply_point_cloud = o3d.data.PLYPointCloud("/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/open3d_data")
#pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
#get_color_map(pcd)



