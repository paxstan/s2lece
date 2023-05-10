import open3d as o3d
import numpy as np
import os
import copy
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


class LidarDataConverter:
    """
    To save point cloud sensor message from ROS Bag to npy files.
    From LiDAR-bonnetal
    """

    def __init__(self, lidar_param, lidar_2_imu, save_dir="data_dir"):
        self.proj_H = lidar_param["height"]
        self.proj_W = lidar_param["width"]
        self.proj_fov_up = lidar_param["v_fov"]["up"]
        self.proj_fov_down = lidar_param["v_fov"]["down"]
        self.translation_l2i = np.array(lidar_2_imu["translation"])
        self.rotation_l2i = np.array(lidar_2_imu["rotation"])
        self.save_dir = save_dir
        self.proj_x = None
        self.proj_y = None
        self.proj_idx = None
        self.proj_xyz = None
        self.proj_mask = None
        self.un_proj_range = None
        self.proj_range = None
        self.points = None
        self.point_idx = None
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        # self.intensity = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.un_proj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def point_cloud_to_np_array(self):
        proj_x, proj_y, depth = project_point_cloud(self.points, height=self.proj_H, width=self.proj_W)
        self.proj_x = np.copy(proj_x)  # store a copy in orig order
        self.proj_y = np.copy(proj_y)  # store a copy in original order
        self.un_proj_range = np.copy(depth)  # copy of depth in original order

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assigning to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    def __call__(self, org_data, num, current_pose, next_pose):
        self.points, timestamp_data = np.hsplit(np.array(org_data), [3])
        timestamp_data = timestamp_data.reshape(-1)

        self.point_cloud_to_np_array()
        world_points = convert_to_world_frame(
            self.points, timestamp_data, current_pose, next_pose, self.rotation_l2i, self.translation_l2i)
        save_dir = os.path.join(self.save_dir, str(num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "idx.npy"), self.proj_idx)
        np.save(os.path.join(save_dir, "world_frame.npy"), np.array(world_points))
        np.save(os.path.join(save_dir, "range.npy"), self.proj_range)
        np.save(os.path.join(save_dir, "valid_mask.npy"), self.proj_mask)
        np.save(os.path.join(save_dir, "xyz.npy"), self.points)
        self.reset()


class Imu2World:
    def __init__(self, points, pose1, pose2, timestamp):
        self.points = points
        self.pose1 = pose1
        self.pose2 = pose2
        self.timestamp = timestamp
        self.x_new = self.time_interpolation()
        self.interp_trans = self.translation_interpolate()
        self.interp_rot = self.rotation_interpolate()

    def transform_pcd(self):
        for i in range(len(self.points)):
            rot = Rotation.from_quat(self.interp_rot[i])
            self.points[i] = rot.apply(self.points[i]) + self.interp_trans[i]

        return self.points

    def interpolate_gt(self):
        self.interp_trans = self.translation_interpolate()
        self.interp_rot = self.rotation_interpolate()
        interp_time = np.expand_dims(np.copy(self.x_new) + self.pose1[0], axis=1)

    def time_interpolation(self):
        inter_f = interp1d([self.pose1[0], self.pose2[0]], [0, 1])
        return inter_f(self.timestamp)

    def rotation_interpolate(self):
        # Convert the rotation matrices to quaternions
        quats = np.vstack([self.pose1[4:], self.pose2[4:]])
        slerp = Slerp(np.arange(len(quats)), Rotation.from_quat(quats))
        interpolated_rot = slerp(self.x_new).as_quat()
        return interpolated_rot

    def translation_interpolate(self):
        translations = np.vstack([self.pose1[1:4], self.pose2[1:4]])
        # linear interpolation function for x, y, and z components
        interp_f = interp1d(np.arange(len(translations)), translations, axis=0, kind='linear')
        # interpolated x, y, and z components
        trans_interp = interp_f(self.x_new)
        return trans_interp


def project_point_cloud(points, height=32, width=1024):
    """ Projects 3D points from a 360° horizontal scan to a 2D image plane."""

    fov_up = 15
    fov_down = -16
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]

    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in ra

    depth = np.linalg.norm(points, 2, axis=1)

    # get angles of all points
    yaw = -np.arctan2(y_lidar, x_lidar)
    pitch = np.arcsin(z_lidar / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= width  # in [0.0, W]
    proj_y *= height  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(width - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(height - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    return proj_x, proj_y, depth


def convert_to_world_frame(points, timestamp_data, current_pose, next_pose, rotation, translation):
    l_pcd = o3d.geometry.PointCloud()
    l_pcd.points = o3d.utility.Vector3dVector(points)
    # transform from lidar frame to imu frame
    i_pcd = pcd_transformation(copy.deepcopy(l_pcd), rotation, translation)

    imu2world = Imu2World(copy.deepcopy(i_pcd.points), current_pose, next_pose, timestamp_data)
    world_points = imu2world.transform_pcd()
    return world_points


def perform_kdtree(points1, points2, threshold=np.inf):
    # Build a KD-tree for points2
    tree = cKDTree(points2)

    # Find the nearest neighbor of each point in points1
    return tree.query(points1, distance_upper_bound=threshold)


def pcd_transformation(pcd, rotation, translation):
    pcd.rotate(pcd.get_rotation_matrix_from_quaternion(rotation=rotation))
    pcd.translate(translation=translation)
    return pcd
