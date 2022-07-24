import open3d as o3d
from pyntcloud import PyntCloud

def get_points_array(path):
    my_point_cloud = PyntCloud.from_file(path)
    points = my_point_cloud.points
    points = points.values

    return points

def visualize(points_arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_arr)
    o3d.visualization.draw_geometries([pcd])
