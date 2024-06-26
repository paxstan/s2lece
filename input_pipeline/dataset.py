import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tvf

to_tensor = tvf.Compose([tvf.ToTensor()])


class LidarBase(Dataset):
    """
    Base class for LidarDataset
    """
    def __init__(self, root):
        Dataset.__init__(self)
        self.root = root

    @staticmethod
    def load_np_file(path):
        return np.load(path)


class RealPairDataset(LidarBase):
    """
    Class to load dataset with correspondences
    """
    def __init__(self, dataset):
        LidarBase.__init__(self, dataset["data_dir"])
        self.root = dataset["data_dir"]
        self.gt_pose = np.load(f'{self.root}/ground_truth_pose.npy') if os.path.exists(f'{self.root}/ground_truth_pose.npy') \
            else None
        self.arr_corres = np.load(f'{self.root}/corres.npy', allow_pickle=True) if os.path.exists(
            f'{self.root}/corres.npy') else None
        self.npairs = self.arr_corres.shape[0]
        self.sensor_img_means = dataset["mean"]
        self.sensor_img_stds = dataset["std"]

    def __len__(self):
        self.npairs = self.arr_corres.shape[0]
        return self.npairs

    def __getitem__(self, idx):
        source = str(self.arr_corres[idx][0])
        target = str(self.arr_corres[idx][1])
        corres_dir = self.arr_corres[idx][2]

        img1 = self.load_np_file(os.path.join(self.root, source, 'range.npy'))
        img2 = self.load_np_file(os.path.join(self.root, target, 'range.npy'))
        mask1 = self.load_np_file(os.path.join(self.root, source, 'valid_mask.npy'))
        mask2 = self.load_np_file(os.path.join(self.root, target, 'valid_mask.npy'))
        idx1 = self.load_np_file(os.path.join(self.root, source, 'idx.npy'))
        idx2 = self.load_np_file(os.path.join(self.root, target, 'idx.npy'))
        xyz1 = self.load_np_file(os.path.join(self.root, source, 'un_proj_xyz.npy'))
        xyz2 = self.load_np_file(os.path.join(self.root, target, 'un_proj_xyz.npy'))

        flow = self.load_np_file(os.path.join(self.root, "correspondence", corres_dir, 'flow.npy'))
        initial_flow = self.load_np_file(os.path.join(self.root, source, 'initial_flow.npy'))

        # img1 = (img1 - self.sensor_img_means) / self.sensor_img_stds
        img1 = img1 * mask1
        img1[img1 == -0.0] = 0.0

        # img2 = (img2 - self.sensor_img_means) / self.sensor_img_stds
        img2 = img2 * mask2
        img2[img2 == -0.0] = 0.0

        mask = (mask1 * mask2).astype(bool)

        result = dict(
            img1=torch.unsqueeze(torch.from_numpy(img1), 0).to(torch.float32),
            img2=torch.unsqueeze(torch.from_numpy(img2), 0).to(torch.float32),
            flow=torch.from_numpy(flow),
            initial_flow=torch.from_numpy(initial_flow),
            mask1=torch.unsqueeze(torch.from_numpy(mask1), 0),
            mask2=torch.unsqueeze(torch.from_numpy(mask2), 0),
            mask=torch.unsqueeze(torch.from_numpy(mask), 0),
            idx1=torch.unsqueeze(torch.from_numpy(idx1), 0),
            idx2=torch.unsqueeze(torch.from_numpy(idx2), 0),
            xyz1=torch.unsqueeze(torch.from_numpy(xyz1), 0),
            xyz2=torch.unsqueeze(torch.from_numpy(xyz2), 0),
            path=corres_dir
        )

        return result


class SingleDataset(LidarBase):
    """
    Class for loading individual scan images
    """
    def __init__(self, dataset):
        LidarBase.__init__(self, dataset["data_dir"])
        self.root = dataset["data_dir"]
        self.npairs = self.get_count()
        self.sensor_img_means = dataset["mean"]
        self.sensor_img_stds = dataset["std"]

    def get_count(self):
        abspath = os.path.abspath(self.root)
        listdir = os.listdir(abspath)
        return sum([os.path.isdir(os.path.join(abspath + "/" + dr)) for dr in listdir if dr != "correspondence"])

    def __len__(self):
        return self.npairs

    def __getitem__(self, idx):
        img = self.load_np_file(os.path.join(self.root, str(idx), 'range.npy'))
        mask = self.load_np_file(os.path.join(self.root, str(idx), 'valid_mask.npy'))

        # img = (img - self.sensor_img_means) / self.sensor_img_stds
        img = img * mask
        img[img == -0.0] = 0.0

        result = dict(
            img=torch.unsqueeze(torch.from_numpy(img), 0).to(torch.float32),
            mask=torch.unsqueeze(torch.from_numpy(mask), 0)
        )
        return result
