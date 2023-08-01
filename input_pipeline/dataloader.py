import numpy as np
import pdb
from PIL import Image
from utils.transform_tools import persp_apply
import torch
import torchvision.transforms as tvf
from torch.utils.data import DataLoader, random_split
from input_pipeline.preprocessing import preprocess_range_image


# # img_mean = [5.4289184]
# # img_std = [9.20105]
# img_mean = [0.02]
# img_std = [0.05]
#
# # normalize_img = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=img_mean, std=img_std)]

# # normalize_img = tvf.Compose([tvf.ToTensor()])

class ToFloat32(object):
    def __call__(self, tensor):
        return tensor.to(torch.float32)


normalize_img = tvf.Compose([tvf.ToTensor(), ToFloat32()])


class PairLoader:
    def __init__(self, dataset):
        assert hasattr(dataset, 'npairs')
        assert hasattr(dataset, 'get_pair')
        self.dataset = dataset

    def __len__(self):
        assert len(self.dataset) == self.dataset.npairs, "not same length"  # and not nimg
        return len(self.dataset)

    def __getitem__(self, i):
        # print(f"item:{i}")
        img_a, img_b, metadata = self.dataset.get_pair(i)
        img_a, edge_weight_a = preprocess_range_image(img_a)
        img_b, edge_weight_a = preprocess_range_image(img_b)
        flow = np.float32(metadata['flow'])
        initial_flow = np.float32(metadata['initial_flow'])
        # flow_mask = metadata.get('flow_mask', np.ones(aflow.shape[:2], np.uint8))
        mask1 = metadata.get('mask1', np.ones(flow.shape[:2], np.uint8))
        mask2 = metadata.get('mask2', np.ones(flow.shape[:2], np.uint8))

        result = dict(
            img1=img_a,
            img2=img_b,
            flow=flow,
            initial_flow=initial_flow,
            mask1=mask1,
            mask2=mask2
        )
        return result


class SingleLoader:
    def __init__(self, dataset):
        assert hasattr(dataset, 'npairs')
        assert hasattr(dataset, 'get_item')
        self.dataset = dataset

    def __len__(self):
        assert len(self.dataset) == self.dataset.npairs, "not same length"  # and not nimg
        return len(self.dataset)

    def __getitem__(self, i):
        # print(f"item:{i}")
        img, mask = self.dataset.get_item(i)
        img, weight = preprocess_range_image(img)

        # if self.distort is not None:
        #     for distort_fn in self.distort:
        #         img = distort_fn(img)
        img = normalize_img(img)

        return {'img': img, 'mask': mask, 'weight': weight}


class oldSingleLoader:
    def __init__(self, dataset):
        assert hasattr(dataset, 'npairs')
        assert hasattr(dataset, 'get_item')
        self.dataset = dataset

    def __len__(self):
        assert len(self.dataset) == self.dataset.npairs, "not same length"  # and not nimg
        return len(self.dataset)

    def __getitem__(self, i):
        print(len(self.dataset))
        img, mask = self.dataset.get_item(0)
        img, weight = preprocess_range_image(img)
        return {'img': img, 'mask': mask, 'weight': weight}


def threaded_loader(loader, iscuda, threads, batch_size=1, shuffle=True):
    """ Get a data loader, given the dataset and some parameters.

        Parameters
        ----------
        loader : object[i] returns the i-th training example.

        iscuda : bool

        batch_size : int

        threads : int

        shuffle : bool

        Returns
        -------
            a multi-threaded pytorch loader.
        """
    return DataLoader(
        loader,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=threads,
        pin_memory=iscuda)
