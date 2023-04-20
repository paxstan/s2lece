import os
import numpy as np
from utils.transform_tools import persp_apply
from PIL import Image
import torchvision.transforms as tvf

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


class CatDataset(Dataset):
    """ Concatenation of several datasets.
    """

    def __init__(self, *datasets):
        assert len(datasets) >= 1
        self.datasets = datasets
        offsets = [0]
        for db in datasets:
            offsets.append(db.nimg)
        self.offsets = np.cumsum(offsets)
        self.nimg = self.offsets[-1]
        self.root = None

    def which(self, i):
        pos = np.searchsorted(self.offsets, i, side='right') - 1
        assert pos < self.nimg, 'Bad image index %d >= %d' % (i, self.nimg)
        return pos, i - self.offsets[pos]

    def get_key(self, i):
        b, i = self.which(i)
        return self.datasets[b].get_key(i)

    def get_filename(self, i):
        b, i = self.which(i)
        return self.datasets[b].get_filename(i)

    def __repr__(self):
        fmt_str = "CatDataset("
        for db in self.datasets:
            fmt_str += str(db).replace("\n", " ") + ', '
        return fmt_str[:-2] + ')'


class PairDataset(Dataset):
    """ A dataset that serves image pairs with ground-truth pixel correspondences.
    """

    def __init__(self):
        Dataset.__init__(self)
        self.npairs = 0

    def get_filename(self, img_idx, root=None):
        if is_pair(img_idx):  # if img_idx is a pair of indices, we return a pair of filenames
            return tuple(Dataset.get_filename(self, i, root) for i in img_idx)
        return Dataset.get_filename(self, img_idx, root)

    def get_image(self, img_idx):
        if is_pair(img_idx):  # if img_idx is a pair of indices, we return a pair of images
            return tuple(Dataset.get_image(self, i) for i in img_idx)
        return Dataset.get_image(self, img_idx)

    def get_corres_filename(self, pair_idx):
        raise NotImplementedError()

    def get_homography_filename(self, pair_idx):
        raise NotImplementedError()

    def get_flow_filename(self, pair_idx):
        raise NotImplementedError()

    def get_mask_filename(self, pair_idx):
        raise NotImplementedError()

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
        for i in range(self.npairs):
            a, b = self.image_pairs[i]
            fns.add(self.get_filename(a))
            fns.add(self.get_filename(b))
        return fns

    def __len__(self):
        return self.npairs  # size should correspond to the number of pairs, not images

    def __repr__(self):
        res = 'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images,' % self.nimg
        res += ' %d image pairs' % self.npairs
        res += '\n  root: %s...\n' % self.root
        return res

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


class LidarData:
    def __init__(self, path):
        self.org_data = np.load(f'{path}/org_data.npy')
        self.range_data = np.load(f'{path}/range.npy')
        self.xyz_data = np.load(f'{path}/xyz.npy')
        self.valid_mask = np.load(f'{path}/valid_mask.npy')


def is_pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 2:
        return True
    return False