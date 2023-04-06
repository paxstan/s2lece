import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from utils.lidar_data_conversion import LidarData, LidarSynthetic
from utils.transform_tools import persp_apply
import torchvision.transforms as tvf

RGB_mean = [0.5]
RGB_std = [0.125]

norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


class Dataset(object):
    """
    Base class for a dataset. To be overloaded.
    """
    root = ''
    img_dir = ''
    num_img = 0

    def __len__(self):
        return self.num_img

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
        res += '  %d images' % self.num_img
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


def point_cloud_extractor(path, data_dir):
    bag = rosbag.Bag(path)
    lidar_scan = LidarData()
    count = 1
    for topic, msg, t in bag.read_messages(topics=['/hesai/pandar']):
        cloud_data = list(pc2.read_points(msg, skip_nans=True, field_names=['x', 'y', 'z', 'intensity']))
        points = np.array(cloud_data)
        lidar_scan.point_cloud_to_np_array(points)
        save_dir = os.path.join(data_dir, str(count))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "range.npy"), lidar_scan.proj_range)
        np.save(os.path.join(save_dir, "intensity.npy"), lidar_scan.proj_intensity)
        np.save(os.path.join(save_dir, "valid_mask.npy"), lidar_scan.proj_mask)
        np.save(os.path.join(save_dir, "xyz.npy"), lidar_scan.proj_xyz)
        count = count + 1
    bag.close()


def is_pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 2:
        return True
    return False
