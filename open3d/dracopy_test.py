import os
import DracoPy
import numpy as np
import open3d as o3d

pcd_object = o3d.io.read_point_cloud("copy_of_fragment.pcd")
print(len(pcd_object.points))
# If faces is omitted, DracoPy will encode a point cloud
binary = DracoPy.encode(pcd_object.points)
with open('bunny_test.drc', 'wb') as test_file:
  test_file.write(binary)


with open('bunny_test.drc', 'rb') as test_file:
  file_content = test_file.read()
  pcd_object = DracoPy.decode_buffer_to_mesh(file_content)
  print('number of points in test file: {0}'.format((pcd_object.points)))
  arr = np.asarray(pcd_object.points)
  print(arr)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(arr)
  o3d.visualization.draw_geometries([pcd])

# Note: If mesh.points is an integer numpy array,
# it will be encoded as an integer attribute. Otherwise,
# it will be encoded as floating point.
# binary = DracoPy.encode(mesh.points, mesh.faces)



# Options for encoding:
# binary = DracoPy.encode(
#   mesh.points, faces=mesh.faces,
#   quantization_bits=14, compression_level=1,
#   quantization_range=-1, quantization_origin=None,
#   create_metadata=False, preserve_order=False
# )
