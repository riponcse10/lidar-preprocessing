import os
import numpy as np
import pickle

def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'r') as f:
        data = pickle.load(f)

    return data


def write_pickle(results, file_path):
    with open(file_path, 'w') as f:
        pickle.dump(results, f)


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def read_calib(file_path, extend_matrix=True):
    with open(file_path, 'r' ) as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    p0 = np.array([lines[0].split(" ")[1:]]).reshape(3, 4).astype(np.float)
    p1 = np.array([lines[1].split(" ")[1:]]).reshape(3, 4).astype(np.float)
    p2 = np.array([lines[2].split(" ")[1:]]).reshape(3, 4).astype(np.float)
    p3 = np.array([lines[3].split(" ")[1:]]).reshape(3, 4).astype(np.float)

    R0_rect = np.array([lines[4].split(" ")[1:]]).reshape(3,3).astype(np.float)
    Tr_velo_to_cam = np.array([lines[5].split(" ")[1:]]).reshape(3,4).astype(np.float)
    Tr_imu_to_velo = np.array([lines[6].split(" ")[1:]]).reshape(3,4).astype(np.float)

    #Why are we interested to extend the matrix???
    if extend_matrix:
        p0 = np.concatenate([p0, np.array([[0, 0, 0, 1]])], axis=0)
        p1 = np.concatenate([p1, np.array([[0, 0, 0, 1]])], axis=0)
        p2 = np.concatenate([p2, np.array([[0, 0, 0, 1]])], axis=0)
        p3 = np.concatenate([p3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict = dict(
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo
    )

    return calib_dict

def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    names, truncated, occluded, alpha, bbox, \
    dimensions, location, rotation_y = [], [], [], [], [], [], [], []
    for line in lines:
        parts = line.split(" ")
        names.append(parts[0])
        truncated.append(parts[1])
        occluded.append(parts[2])
        alpha.append(parts[3])
        bbox.append(parts[4:8])
        dimensions.append(parts[8:11])
        location.append(parts[11:14])
        rotation_y.append(parts[14])

    annotation = {}
    #annotation['name'] = np.array(names)
    annotation['truncated'] = np.array(truncated).astype(np.float)
    annotation['occluded'] = np.array(occluded).astype(np.int)
    annotation['alpha'] = np.array(alpha).astype(np.float)
    annotation['bbox'] = np.array(bbox).astype(np.float)
    annotation['dimensions'] = np.array(bbox).astype(np.float)[:, [2, 0, 1]]
    annotation['location'] = np.array(location).astype(np.float)
    annotation['rotation_y'] = np.array(rotation_y).astype(np.float)

    return annotation

def bbox_camera2lidar(bboxes, tr_velo2cam, r0_rect):
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([x_size, y_size, z_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo2cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)

    return np.array(bboxes_lidar, dtype=np.float)


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |
    y      | /              |  |     | |
    <------|o               | 7 -----| 4
                            |/   o   |/
                            3 ------ 0
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]],
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]],
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners