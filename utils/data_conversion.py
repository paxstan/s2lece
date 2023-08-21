import open3d as o3d
import numpy as np
import os
import copy
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from visualization.visualization import flow_to_color


class LidarDataConverter:
    """
    To save point cloud sensor message from ROS Bag to npy files.
    From LiDAR-bonnetal
    """

    def __init__(self, lidar_param, lidar_type, save_dir="data_dir", generate_gt=True):
        self.generate_gt = generate_gt
        self.proj_H = lidar_param["height"]
        self.proj_W = lidar_param["width"]
        self.proj_fov_up = lidar_param["v_fov"]["up"]
        self.proj_fov_down = lidar_param["v_fov"]["down"]
        self.translation_l2i = np.array(lidar_param["lidar_to_imu"]["translation"])
        self.rotation_l2i = np.array(lidar_param["lidar_to_imu"]["rotation"])
        self.lidar_type = lidar_type
        self.save_dir = save_dir
        self.proj_x = None
        self.proj_y = None
        self.proj_idx = None
        self.un_proj_xyz = None
        self.proj_xyz = None
        self.proj_mask = None
        self.un_proj_range = None
        self.proj_range = None
        self.points = None
        self.point_idx = None
        self.initial_flow = None
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        # self.intensity = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.un_proj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        self.un_proj_xyz = np.zeros((0, 1), dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask

        self.initial_flow = np.zeros((2, self.proj_H, self.proj_W))

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def point_cloud_to_np_array(self):
        proj_x, proj_y, depth = project_point_cloud(self.points,
                                                    height=self.proj_H, width=self.proj_W,
                                                    fov_up=self.proj_fov_up, fov_down=self.proj_fov_down)
        self.proj_x = np.copy(proj_x)  # store a copy in orig order
        self.proj_y = np.copy(proj_y)  # store a copy in original order
        self.un_proj_range = np.copy(depth)
        self.un_proj_xyz = np.copy(self.points)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]

        # assigning to images
        self.proj_range[proj_x, proj_y] = depth
        self.proj_xyz[proj_x, proj_y] = points
        self.proj_idx[proj_x, proj_y] = indices
        self.proj_mask[proj_x, proj_y] = (self.proj_idx[proj_x, proj_y] != -1).astype(np.int32)
        self.initial_flow[0, proj_x, proj_y] = proj_x
        self.initial_flow[1, proj_x, proj_y] = proj_y

    def __call__(self, org_data, num, current_pose=None, next_pose=None):
        self.points, timestamp_data = np.hsplit(np.array(org_data), [3])
        timestamp_data = timestamp_data.reshape(-1)

        self.point_cloud_to_np_array()
        save_dir = os.path.join(self.save_dir, str(num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "proj_x.npy"), self.proj_x)
        np.save(os.path.join(save_dir, "proj_y.npy"), self.proj_y)
        np.save(os.path.join(save_dir, "idx.npy"), self.proj_idx)
        np.save(os.path.join(save_dir, "range.npy"), self.proj_range)
        np.save(os.path.join(save_dir, "un_proj_range.npy"), self.un_proj_range)
        np.save(os.path.join(save_dir, "valid_mask.npy"), self.proj_mask)
        np.save(os.path.join(save_dir, "proj_xyz_sorted.npy"), self.points)
        np.save(os.path.join(save_dir, "xyz.npy"), self.proj_xyz)
        np.save(os.path.join(save_dir, "un_proj_xyz.npy"), self.un_proj_xyz)
        np.save(os.path.join(save_dir, "initial_flow.npy"), self.initial_flow)
        if self.generate_gt:
            world_points = convert_to_world_frame(
                self.points, timestamp_data,
                self.rotation_l2i, self.translation_l2i, self.lidar_type,
                current_pose, next_pose
            )
            np.save(os.path.join(save_dir, "world_frame.npy"), np.array(world_points))
            w_frame = o3d.geometry.PointCloud()
            w_frame.points = o3d.utility.Vector3dVector(world_points)
            o3d.io.write_point_cloud(os.path.join(save_dir, "world_frame.pcd"), w_frame)
        self.reset()
        return save_dir


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


def project_point_cloud(points, height=32, width=1024, fov_up=15, fov_down=-16):
    """ Projects 3D points from a 360° horizontal scan to a 2D image plane."""

    # fov_up = 15
    # fov_down = -16
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
    proj_x = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    proj_y = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= height  # in [0.0, H]
    proj_y *= width  # in [0.0, W]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(height - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,H-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(width - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,W-1]

    return proj_x, proj_y, depth


def convert_to_world_frame(points, timestamp_data, rotation, translation, lidar_type, current_pose, next_pose=None):
    world_points = None
    if lidar_type == "hilti":
        l_pcd = o3d.geometry.PointCloud()
        l_pcd.points = o3d.utility.Vector3dVector(points)
        # transform from lidar frame to imu frame
        i_pcd = pcd_transformation(copy.deepcopy(l_pcd), rotation, translation)
        imu2world = Imu2World(copy.deepcopy(i_pcd.points), current_pose, next_pose, timestamp_data)
        world_points = imu2world.transform_pcd()
    elif lidar_type == "kitti":
        pose_translation = current_pose[1:4]
        pose_rotation = current_pose[4:]
        l_pcd = o3d.geometry.PointCloud()
        l_pcd.points = o3d.utility.Vector3dVector(points)
        # transform from lidar frame to imu frame
        i_pcd = pcd_transformation(copy.deepcopy(l_pcd), pose_rotation, pose_translation)
        world_points = i_pcd.points

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


def get_pixel_match(source, target, nearest_distance, height, width):
    csv_dict = dict()
    source_frame = np.load(os.path.join(source, 'world_frame.npy'))
    target_frame = np.load(os.path.join(target, 'world_frame.npy'))

    source_idx = np.load(os.path.join(source, 'idx.npy'))
    target_idx = np.load(os.path.join(target, 'idx.npy'))

    source_mask = np.load(os.path.join(source, 'valid_mask.npy'))
    target_mask = np.load(os.path.join(target, 'valid_mask.npy'))

    source_x_y = map_points_xy(source_idx, source_mask, source_frame.shape[0])
    target_x_y = map_points_xy(target_idx, target_mask, target_frame.shape[0])

    distances, corres = perform_kdtree(source_frame, target_frame, threshold=nearest_distance)
    csv_dict["correspondence"] = len(corres)

    # count = np.bincount(corres)
    #
    # true_val = np.where(count == 1)[0]
    # true_indices = [np.column_stack(np.where(corres == i)).ravel().tolist() for i in true_val]
    # true_indices = np.array(true_indices)
    # unique_mask = np.zeros_like(corres)
    # unique_mask[true_indices] = 1
    #
    # non_unique = np.where(count > 1)[0]
    # non_unique_pair = [[i, np.column_stack(np.where(corres == i)).ravel().tolist()] for i in non_unique]
    # for i, val in enumerate(non_unique_pair):
    #     min_distance_id = distances[val[1]].argmin()
    #     unique_mask[val[1][min_distance_id]] = 1
    #
    # unique_mask = unique_mask.astype(bool)
    #
    neighbour_mask = distances != np.inf

    mask = neighbour_mask  # * unique_mask

    csv_dict["neighbours"] = np.count_nonzero(neighbour_mask.astype(int))
    # csv_dict["unique"] = np.count_nonzero(unique_mask.astype(int))
    csv_dict["valid (neighbor and unique)"] = np.count_nonzero(mask.astype(int))

    flow, flow_img = build_flow(source_x_y, target_x_y, corres, mask, height, width)

    pixel_count = flow.shape[1] * flow.shape[2]
    csv_dict[f"non zero flow along x (out of {pixel_count} pixels)"] = np.count_nonzero(flow[0])
    csv_dict[f"non zero flow along x in percent"] = round(((np.count_nonzero(flow[0]) / pixel_count) * 100), 4)
    csv_dict[f"non zero flow along y (out of {pixel_count} pixels)"] = np.count_nonzero(flow[1])
    csv_dict[f"non zero flow y in percent"] = round(((np.count_nonzero(flow[1]) / pixel_count) * 100), 4)

    return flow, flow_img, [csv_dict]


def map_points_xy(idx, valid_mask, no_of_points):
    source_x_y = np.full((no_of_points, 2), np.nan)
    # valid_idx = np.argwhere(np.invert(valid_mask))
    # valid_val = idx[valid_idx[:, 0], valid_idx[:, 1]]
    # sort_id = np.argsort(valid_val)
    # source_x_y[valid_val[sort_id]] = valid_idx[sort_id]
    valid_idx = np.where(idx != -1)
    valid_val = idx[valid_idx[0], valid_idx[1]]
    source_x_y[valid_val, 0] = valid_idx[0]
    source_x_y[valid_val, 1] = valid_idx[1]
    return source_x_y


def build_flow(source_idx, target_idx, indices, n_mask, height, width):
    s_idx = source_idx[n_mask]
    # s_flow = np.zeros((2, 32, 2000))
    # s_flow[0, x, y] = x
    # s_flow[1, x, y] = y

    t_idx = target_idx[indices[n_mask]]

    diff_x = t_idx[:, 0] - s_idx[:, 0]
    diff_y = t_idx[:, 1] - s_idx[:, 1]

    # mask = (np.isnan(diff_x)) | (np.isnan(diff_y)) | np.invert(n_mask)
    # s_idx = s_idx[np.invert(mask)].astype(int)
    # diff_x = diff_x[np.invert(mask)].astype(int)
    # diff_y = diff_y[np.invert(mask)].astype(int)
    mask = np.invert(np.isnan(diff_x)) & np.invert(np.isnan(diff_y))
    s_idx = s_idx[mask].astype(int)
    diff_x = diff_x[mask].astype(int)
    diff_y = diff_y[mask].astype(int)

    x = s_idx[:, 0]
    y = s_idx[:, 1]

    flow = np.zeros((2, height, width))
    flow[0, x, y] = diff_x
    flow[1, x, y] = diff_y

    flow_img = flow_to_color(flow.transpose(1, 2, 0))

    return flow, flow_img


def synth_flow(source, target):
    source_xyz = np.load(os.path.join(source, 'un_proj_xyz.npy'))
    target_xyz = np.load(os.path.join(target, 'un_proj_xyz.npy'))

    source_idx = np.load(os.path.join(source, 'idx.npy'))
    target_idx = np.load(os.path.join(target, 'idx.npy'))

    source_mask = np.load(os.path.join(source, 'valid_mask.npy'))
    target_mask = np.load(os.path.join(target, 'valid_mask.npy'))

    corres = np.arange(source_xyz.shape[0])

    source_x_y = map_points_xy(source_idx, source_mask, source_xyz.shape[0])
    target_x_y = map_points_xy(target_idx, target_mask, target_xyz.shape[0])

    # mask = source_mask * target_mask
    mask = np.ones_like(corres).astype(bool)

    # corres = corres[mask]

    flow = build_flow(source_x_y, target_x_y, corres, mask)

    flow_img = flow_to_color(flow.transpose(1, 2, 0))

    return flow, flow_img
