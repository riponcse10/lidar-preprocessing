import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def get_color_map(pcd):
    #pcd = o3d.io.read_point_cloud(filename="copy_of_fragment.pcd")


    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    print(labels)
    print(colors)
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])


pcd = o3d.io.read_point_cloud(filename="copy_of_fragment.pcd")
print(pcd.points[1500])
get_color_map(pcd)

#ply_point_cloud = o3d.data.PLYPointCloud("/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/open3d_data")
#pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
#get_color_map(pcd)



