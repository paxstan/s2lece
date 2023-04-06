"""
reference taken from 3D3L and LiDAR-bonnetal
"""
import numpy as np
import os
from PIL import Image
import random

H_RES = 0.18  # horizontal resolution (10Hz setting)
V_RES = 1  # vertical res
V_FOV = (-16, 15)  # Field of view (-ve, +ve) along vertical axis

class LidarData:
    """
    From LiDAR-bonnetal
    """

    def __init__(self, h=32, w=1024, fov_up=15, fov_down=-16):
        self.proj_mask = None
        self.proj_y = None
        self.proj_x = None
        self.proj_idx = None
        self.proj_intensity = None
        self.proj_xyz = None
        self.un_proj_range = None
        self.proj_range = None
        self.intensity = None
        self.points = None
        self.proj_H = h
        self.proj_W = w
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
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

    def point_cloud_to_np_array(self, points_org):
        self.points = points_org[:, 0:3]
        x_lidar = self.points[:, 0]
        y_lidar = self.points[:, 1]
        z_lidar = self.points[:, 2]
        self.intensity = points_org[:, 3]

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
        remission = self.intensity[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_intensity[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    def point_cloud_to_np_array_two(self, points_org, v_res=V_RES, h_res=H_RES, v_fov=V_FOV):
        self.points = points_org[:, 0:3]
        x_lidar = self.points[:, 0]
        y_lidar = self.points[:, 1]
        z_lidar = self.points[:, 2]
        self.intensity = points_org[:, 3]

        # Distance relative to origin when looked from top
        d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)

        v_fov_total = -v_fov[0] + v_fov[1]

        # PROJECT INTO IMAGE COORDINATES
        x_img = np.arctan2(y_lidar, x_lidar)
        y_img = np.arctan2(z_lidar, d_lidar)
        pixel_values = np.sqrt(x_lidar ** 2 + y_lidar ** 2 + z_lidar ** 2)

        # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
        x_min = - np.pi  # Theoretical min x value based on sensor specs
        x_img -= x_min  # Shift
        x_max = np.pi  # Theoretical max x value after shifting

        y_min = v_fov[0] * np.pi / 180  # theoretical min y value based on sensor specs
        y_img -= y_min  # Shift
        y_max = v_fov_total * np.pi / 180  # Theoretical max x value after shifting

        # get pixel coordinates
        x_img /= h_res * np.pi / 180
        y_img /= v_res * np.pi / 180

        x_max /= h_res * np.pi / 180
        y_max /= v_res * np.pi / 180

        dpi = 100

        self.proj_range = np.full((x_max / dpi, self.proj_W), -1,
                                  dtype=np.float32)

        self.proj_range[y_img, x_img] = pixel_values


class LidarSynthetic:
    def __init__(self, root, skip=(0, -1, 1), crop=False, crop_size=0):
        self.img_folders = os.listdir(root)[skip[0]: skip[1]: skip[2]]
        self.img_folders.sort()
        self.num_img = len(self.img_folders)
        self.root = root
        self.crop = crop
        self.crop_size = crop_size

    @staticmethod
    def open_npy(path):
        return np.load(path)

    def get_xyz(self, idx):
        folder_path = os.path.join(self.root, self.img_folders[idx])
        xyz = self.open_npy(folder_path + '/xyz.npy')
        return xyz

    def get_valid_range_mask(self, idx):
        folder_path = os.path.join(self.root, self.img_folders[idx])
        mask = self.open_npy(folder_path + '/valid_mask.npy')
        return mask

    def open_lidar_image(self, folder_path):
        # convert to image
        def img_conv(m):
            im = ((m / 8. + 1.) / 2. * 255).astype(np.uint8)
            return im

        org_range = self.open_npy(folder_path + '/range.npy')
        range_l = img_conv(self.open_npy(folder_path + '/range.npy'))
        # reflectivity_l = img_conv(self.open_npy(folder_path + '/reflectivity.npy'))
        org_intensity = self.open_npy(folder_path + '/intensity.npy')
        intensity_l = img_conv(self.open_npy(folder_path + '/intensity.npy'))

        # img = np.stack((range_l, intensity_l, reflectivity_l), axis=2)
        img = np.stack((range_l, intensity_l), axis=2)
        img = Image.fromarray(range_l)

        return img

    def get_image(self, img_idx):
        folder_path = os.path.join(self.root, self.img_folders[img_idx])
        img = self.open_lidar_image(folder_path)
        # img = np.array(img)
        # img[~self.get_valid_range_mask(img_idx), :] = 255
        # img = Image.fromarray(img)
        # crop image to avoid statically undefined areas
        # if self.type == 'OS0' and self.crop:
        #     regions = np.array([[91, 351], [604, 864]])
        #     region = int(random.random() < 0.5)
        #     w, h = img.size
        #     shape = (regions[region, 0], 0, regions[region, 1], h)
        #     img = img.crop(shape)
        w, h = img.size
        if self.crop:
            x = random.randint(0, w - self.crop_size)
            img = img.crop((x, 0, x + self.crop_size, h))
        return img



