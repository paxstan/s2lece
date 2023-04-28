import os
import numpy as np
from utils.transform_tools import persp_apply
from utils.data_conversion import project_point_cloud
from visualization.visualization import flow_to_color
import matplotlib.pyplot as pl
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torchvision.transforms as tvf
from scipy.signal import convolve2d
from scipy.spatial import KDTree

RGB_mean = [0.5]
RGB_std = [0.125]
norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


class Dataset(object):
    """ Base class for a dataset. To be overloaded.
    """
    root = ''
    img_dir = ''
    nimg = 0

    def __len__(self):
        return self.nimg

    def get_key(self, img_idx):
        raise NotImplementedError()

    def get_filename(self, img_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_key(img_idx))

    def get_image(self, img_idx):
        from PIL import Image
        fname = self.get_filename(img_idx)
        try:
            return Image.open(fname).convert('RGB')
        except Exception as e:
            raise IOError("Could not load image %s (reason: %s)" % (fname, str(e)))

    def __repr__(self):
        res = 'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images' % self.nimg
        res += '\n  root: %s...\n' % self.root
        return res


class SyntheticPair:
    """ A synthetic generator of image pairs.
            Given a normal image dataset, it constructs pairs using random homographies & noise.
        """

    def __init__(self, scale, distort):
        self.distort = distort
        self.scale = scale

    @staticmethod
    def make_pair(img):
        return img, img

    def get_pair(self, org_img, output=('aflow')):
        """ Procedure:
        This function applies a series of random transformations to one original image
        to form a synthetic image pairs with perfect ground-truth.
        """
        if isinstance(output, str):
            output = output.split()

        original_img = org_img
        scaled_image = self.scale(original_img)
        scaled_image, scaled_image2 = self.make_pair(scaled_image['img'])
        # scaled_image = original_img
        rand_tilt = self.distort[0](inp=dict(img=scaled_image2, persp=(1, 0, 0, 0, 1, 0, 0, 0)))
        rand_noise = self.distort[1](inp=rand_tilt)
        scaled_and_distorted_image = self.distort[2](inp=rand_noise)
        # scaled_and_distorted_image = self.distort(
        #     dict(img=scaled_image2, persp=(1, 0, 0, 0, 1, 0, 0, 0)))
        W, H = scaled_image.size
        trf = scaled_and_distorted_image['persp']

        meta = dict()
        meta['mask'] = org_img['mask']
        if 'aflow' in output or 'flow' in output:
            # compute optical flow
            xy = np.mgrid[0:H, 0:W][::-1].reshape(2, H * W).T
            aflow = np.float32(persp_apply(trf, xy).reshape(H, W, 2))
            meta['flow'] = aflow - xy.reshape(H, W, 2)
            meta['aflow'] = aflow

        if 'homography' in output:
            meta['homography'] = np.float32(trf + (1,)).reshape(3, 3)

        return scaled_image, scaled_and_distorted_image['img'], meta


class LidarBase(Dataset):
    def __init__(self, crop=True):
        Dataset.__init__(self)
        self.crop = crop
        self.type = type
        self.crop_size = 250
        self.H = 128

    def open_lidar_image(self, folder_path):
        rang = self.open_img(folder_path + '/range.npy')
        img = Image.fromarray(rang)
        return img

    @staticmethod
    def open_img(path):
        return np.load(path)


class LidarPairDataset(LidarBase):
    def __init__(self, root, crop=False, reproject=True):
        LidarBase.__init__(self, crop)
        self.root = root
        self.gt_pose = np.load(f'{root}/ground_truth_pose.npy') if os.path.exists(f'{root}/ground_truth_pose.npy') \
            else None
        self.arr_corres = np.load(f'{root}/corres.npy', allow_pickle=True) if os.path.exists(
            f'{root}/corres.npy') else None
        self.npairs = self.arr_corres.shape[0]
        self.crop_size = 150
        self.reproject = reproject

    def __len__(self):
        self.npairs = self.arr_corres.shape[0]
        return self.npairs

    @staticmethod
    def get_homog_matrix(pose):
        """
        Transforms a pose to a homogeneous matrix
        :param pose: [x, y, z, qx, qy, qz, qw]
        :return: 4x4 homogeneous matrix
        """
        m = np.eye(4)
        rot = R.from_quat(pose[3:])
        for i in range(3):
            m[i, 3] = pose[i]
        m[0:3, 0:3] = rot.as_matrix()
        return m

    def get_pair(self, idx, output=()):
        """ returns (img1, img2, `metadata`)
        """
        source = str(self.arr_corres[idx][0])
        target = str(self.arr_corres[idx][1])
        corres = self.arr_corres[idx][2]

        img1 = np.array(self.open_lidar_image(os.path.join(self.root, source)))
        img2 = np.array(self.open_lidar_image(os.path.join(self.root, target)))
        mask1 = np.load(os.path.join(self.root, source, 'valid_mask.npy')).reshape(-1)
        mask2 = np.load(os.path.join(self.root, target, 'valid_mask.npy')).reshape(-1)
        idx1 = np.load(os.path.join(self.root, source, 'idx.npy'))
        idx2 = np.load(os.path.join(self.root, target, 'idx.npy'))

        h1, w1 = img1.shape
        h2, w2 = img2.shape

        img1 = img1.reshape(-1)
        img2 = img2.reshape(-1)

        img1[np.invert(mask1)] = 0
        img2[np.invert(mask2)] = 0

        # reshape masks of valid pixels to image sizes
        mask1 = mask1.reshape(h1, w1)
        mask2 = mask2.reshape(h2, w2)

        # get flow according to pair index and reprojected mask2
        flow, mask_valid_in_2 = self.get_pixel_match(mask2, source, target)

        # redefine flow mask with invalid pixels in image1 and maks generated by finding flow
        mask = (mask1 * mask_valid_in_2).astype(bool)

        # set flow for invalid pixels to nan, which is ignored during training
        flow[~mask, :] = np.nan

        # crop image
        img2 = Image.fromarray(img2.reshape(h1, w1))
        img1 = Image.fromarray(img1.reshape(h1, w1))

        meta = {'aflow': flow, 'mask': mask}
        return img1, img2, meta

    def get_pixel_match(self, mask2, source, target):
        xyz1 = np.load(os.path.join(self.root, source, 'xyz.npy'))
        xyz2 = np.load(os.path.join(self.root, target, 'xyz.npy'))

        pose1 = self.gt_pose[int(source), 1:]
        pose2 = self.gt_pose[int(target), 1:]
        tf1 = self.get_homog_matrix(pose1)
        tf2 = self.get_homog_matrix(pose2)
        # get relative transformation between img1 and img2
        tf = np.linalg.inv(tf1) @ tf2

        # Make position vectors homogeneous
        pos1 = np.append(xyz1, np.ones_like(xyz1[:, :, 0:1]), axis=2)
        h1, w1, _ = pos1.shape
        h2, w2 = mask2.shape
        # transform all position vectors from img1 to frame2
        pos1 = pos1.reshape((h1 * w1, 4)).transpose((1, 0))
        xyz_t = np.dot(tf, pos1).transpose((1, 0))[:, :3] # ones added in 201 will be removed here

        # get indices of pixel correspondences
        proj_idx_1 = project_point_cloud(xyz_t, height=h2, width=w2)
        proj_idx_1 = np.array(proj_idx_1)
        # create flow matrix needed for training from pixel correspondences
        flow = np.zeros((h1, w1, 2))
        x_img = proj_idx_1[0, :].reshape(h1, w1)
        y_img = proj_idx_1[1, :].reshape(h1, w1)
        flow[:, :, 0] = x_img
        flow[:, :, 1] = y_img

        # get mask of which pixels in image1 are valid in image2 (out of bounds and invalid defined by mask2)
        x_img = flow[:, :, 0].reshape(-1).astype(int)
        y_img = flow[:, :, 1].reshape(-1).astype(int)
        mask_valid_in_2 = np.zeros(h1 * w1, dtype=bool)
        mask_in_bound = (y_img >= 0) * (y_img < h2) * (x_img >= 0) * (x_img < w2)
        mask_valid_in_2[mask_in_bound] = mask2[y_img[mask_in_bound], x_img[mask_in_bound]]

        # get ranges of valid image2 points
        r2 = np.linalg.norm(xyz2[y_img[mask_valid_in_2], x_img[mask_valid_in_2]], axis=1)
        # get corresponding transformed ranges of image1
        r1_t = np.linalg.norm(xyz_t.reshape(-1, 3)[mask_valid_in_2], axis=1)

        # check if points in image1 are occluded in 2
        d = (r2 - r1_t) / r1_t
        # 20% threshold when close to each other
        not_occluded = d > -0.2
        # define occlusion mask
        occlusion_mask = np.ones((h1 * w1), dtype=bool)
        occlusion_mask[mask_valid_in_2] = not_occluded

        # set mask_valid_in_2 false when occluded
        mask_valid_in_2 = np.logical_and(mask_valid_in_2, occlusion_mask)
        mask_valid_in_2 = mask_valid_in_2.reshape(h1, w1)

        # mask flow according to occluded points
        flow[:, :, 0][np.invert(mask_valid_in_2)] = np.nan
        flow[:, :, 1][np.invert(mask_valid_in_2)] = np.nan

        return flow, mask_valid_in_2


class LidarData:
    def __init__(self, path):
        self.corres = np.load(f'{path}/corres.npy') if os.path.exists(f'{path}/corres.npy') else None
        self.idx_data = np.load(f'{path}/idx.npy') if os.path.exists(f'{path}/idx.npy') else None
        self.world_frame = np.load(f'{path}/world_frame.npy') if os.path.exists(f'{path}/world_frame.npy') else None
        self.range_data = np.load(f'{path}/range.npy') if os.path.exists(f'{path}/range.npy') else None
        self.xyz_data = np.load(f'{path}/xyz.npy') if os.path.exists(f'{path}/xyz.npy') else None
        self.valid_mask = np.load(f'{path}/valid_mask.npy') if os.path.exists(f'{path}/valid_mask.npy') else None


def is_pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 2:
        return True
    return False