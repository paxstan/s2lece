import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from PIL import Image
from utils.lidar_data_conversion import LidarData, LidarSynthetic


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


def generate_data(data_dir):
    synth_data = LidarSynthetic(
        data_dir, skip=(0, -1, 10), crop=False)
    img = synth_data.get_image(1)


def is_pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 2:
        return True
    return False


class PairDataset(Dataset):
    """ A dataset that serves image pairs with ground-truth pixel correspondences.
        """

    def __init__(self):
        Dataset.__init__(self)
        self.image_pairs = None
        self.num_pairs = 0

    def __repr__(self):
        res = 'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images,' % self.num_img
        res += ' %d image pairs' % self.num_pairs
        res += '\n  root: %s...\n' % self.root
        return res

    def get_filename(self, img_idx, root=None):
        if is_pair(img_idx):  # if img_idx is a pair of indices, we return a pair of filenames
            return tuple(Dataset.get_filename(self, i, root) for i in img_idx)
        return Dataset.get_filename(self, img_idx, root)

    def get_image(self, img_idx):
        if is_pair(img_idx):  # if img_idx is a pair of indices, we return a pair of images
            return tuple(Dataset.get_image(self, i) for i in img_idx)
        return Dataset.get_image(self, img_idx)

    def get_pair(self, idx, output=()):
        """ returns (img1, img2, `metadata`)

        `metadata` is a dict() that can contain:
            flow: optical flow
            aflow: absolute flow
            corres: list of 2d-2d correspondences
            mask: boolean image of flow validity (in the first image)
            ...
        """
        raise NotImplementedError()

    def get_paired_images(self):
        fns = set()
        for i in range(self.num_pairs):
            a, b = self.image_pairs[i]
            fns.add(self.get_filename(a))
            fns.add(self.get_filename(b))
        return fns

    def get_key(self, img_idx):
        raise NotImplementedError()

    def get_corres_filename(self, pair_idx):
        raise NotImplementedError()

    def get_homography_filename(self, pair_idx):
        raise NotImplementedError()

    def get_flow_filename(self, pair_idx):
        raise NotImplementedError()

    def get_mask_filename(self, pair_idx):
        raise NotImplementedError()

    @staticmethod
    def _flow2png(flow, path):
        flow = np.clip(np.around(16 * flow), -2 ** 15, 2 ** 15 - 1)
        bytes = np.int16(flow).view(np.uint8)
        Image.fromarray(bytes).save(path)
        return flow / 16

    @staticmethod
    def _png2flow(path):
        try:
            flow = np.asarray(Image.open(path)).view(np.int16)
            return np.float32(flow) / 16
        except:
            raise IOError("Error loading flow for %s" % path)


class SyntheticPairDataset (PairDataset):
    """ A synthetic generator of image pairs.
            Given a normal image dataset, it constructs pairs using random homographies & noise.
        """

    def __init__(self, dataset, scale='', distort=''):
        self.attach_dataset(dataset)
        self.distort = instanciate_transformation(distort)
        self.scale = instanciate_transformation(scale)

    def __repr__(self):
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images and pairs' % self.npairs
        res += '\n  root: %s...' % self.dataset.root
        res += '\n  Scale: %s' % (repr(self.scale).replace('\n',''))
        res += '\n  Distort: %s' % (repr(self.distort).replace('\n',''))
        return res + '\n'

    def attach_dataset(self, dataset):
        assert isinstance(dataset, Dataset) and not isinstance(dataset, PairDataset)
        self.dataset = dataset
        self.num_pairs = dataset.num_img
        self.get_image = dataset.get_image
        self.get_key = dataset.get_key
        self.get_filename = dataset.get_filename
        self.root = None

    def make_pair(self, img):
        return img, img

    def get_pair(self, i, output=('aflow')):
        """ Procedure:
        This function applies a series of random transformations to one original image
        to form a synthetic image pairs with perfect ground-truth.
        """




