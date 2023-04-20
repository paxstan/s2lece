import open3d as o3d
import numpy as np
import os
from utils.plot_tools import draw_point_cloud_pair
from utils.point_cloud_registration import PointCloudRegistration
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d

H_RES = 0.18  # horizontal resolution (10Hz setting)
V_RES = 1  # vertical res
V_FOV = (-16, 15)  # Field of view (-ve, +ve) along vertical axis


def pcd_transformation(pcd, rotation, translation):
    pcd.rotate(pcd.get_rotation_matrix_from_quaternion(rotation=rotation))
    pcd.translate(translation=translation)
    return pcd


def rotation_interpolate(rotation_list):
    if len(rotation_list) > 1:
        # alpha is the interpolation parameter, which ranges from 0 to 1
        alpha = 0.5

        # Convert the rotation matrices to quaternions
        rot_matrix = [Rotation.from_matrix(rot).as_matrix() for rot in rotation_list]

        key_times = np.arange(len(rot_matrix))

        slerp = Slerp(key_times, Rotation.from_matrix(rot_matrix))

        interpolated_rot = slerp([alpha]).as_quat()

        print("Interpolated rotation matrix: \n", interpolated_rot)
        return interpolated_rot
    else:
        return Rotation.from_matrix(rotation_list[0]).as_quat()


def translation_interpolate(translation_list):
    translations = np.asarray(translation_list)
    x = translations[:, 0]
    y = translations[:, 1]
    z = translations[:, 2]
    # new x values to interpolate at
    x_new = np.linspace(0, 1, len(translation_list))

    # linear interpolation function for x, y, and z components
    f_x = interp1d(np.arange(len(x)), x, kind='linear')
    f_y = interp1d(np.arange(len(y)), y, kind='linear')
    f_z = interp1d(np.arange(len(z)), z, kind='linear')

    # interpolated x, y, and z components
    x_interp = f_x(x_new)
    y_interp = f_y(x_new)
    z_interp = f_z(x_new)

    # interpolated vectors
    interpolated_translation = np.row_stack((x_interp, y_interp, z_interp))

    return interpolated_translation


def perform_icp(list_of_pcd):
    i = 0
    icp_result_list = []
    icp_reg = PointCloudRegistration()
    while i < len(list_of_pcd) - 1:
        icp_result = icp_reg(list_of_pcd[i], list_of_pcd[i + 1])
        icp_result_list.append(icp_result)
        i = i+1
    return icp_result_list


def interpolated_data(list_of_pcd):
    icp_result_list = perform_icp(list_of_pcd)
    transformation_matrix_list = [result.transformation for result in icp_result_list]
    translation_list = [transform[0:3, 3] for transform in transformation_matrix_list]
    rotation_list = [np.copy(transform[0:3, 0:3]) for transform in transformation_matrix_list]
    for rotation in rotation_list:
        for i in range(3):
            rotation[:, i] = rotation[:, i] / np.linalg.norm(rotation[:, i])

    return rotation_interpolate(rotation_list), translation_interpolate(translation_list)


class LidarDataConverter:
    """
    To save point cloud sensor message from ROS Bag to npy files.
    From LiDAR-bonnetal
    """

    def __init__(self, h=32, w=1024, fov_up=15, fov_down=-16, save_dir="data_dir"):
        self.proj_x = None
        self.proj_y = None
        self.proj_idx = None
        self.proj_xyz = None
        self.proj_mask = None
        self.un_proj_range = None
        self.proj_range = None
        self.points = None
        self.proj_H = h
        self.proj_W = w
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.save_dir = save_dir
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.intensity = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.un_proj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_intensity = np.full((self.proj_H, self.proj_W), -1,
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
        x_lidar = self.points[:, 0]
        y_lidar = self.points[:, 1]
        z_lidar = self.points[:, 2]

        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in ra

        depth = np.linalg.norm(self.points, 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(y_lidar, x_lidar)
        pitch = np.arcsin(z_lidar / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.un_proj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    def __call__(self, org_data, num):
        self.points = np.asarray([data[:3] for data in org_data])
        self.point_cloud_to_np_array()
        save_dir = os.path.join(self.save_dir, str(num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "org_data.npy"), np.asarray(org_data))
        np.save(os.path.join(save_dir, "range.npy"), self.proj_range)
        np.save(os.path.join(save_dir, "valid_mask.npy"), self.proj_mask)
        np.save(os.path.join(save_dir, "xyz.npy"), self.proj_xyz)
        self.reset()
