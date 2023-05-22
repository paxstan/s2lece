import numpy as np
import pdb
from PIL import Image
from utils.transform_tools import persp_apply
import torchvision.transforms as tvf
import torch
from torch.utils.data import DataLoader
from utils.common import patch_extractor

img_mean = [0.5]
img_std = [0.125]
normalize_img = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=img_mean, std=img_std)])


class PairLoader:
    def __init__(self, dataset, crop, scale, distort):
        assert hasattr(dataset, 'npairs')
        assert hasattr(dataset, 'get_pair')
        self.dataset = dataset
        self.crop = crop
        self.scale = scale
        self.distort = distort
        self.n_samples = 5  # number of random trials per image
        self.norm = normalize_img

    def __len__(self):
        assert len(self.dataset) == self.dataset.npairs, "not same length"  # and not nimg
        return len(self.dataset)

    def __getitem__(self, i):
        # Retrieve an image pair and their absolute flow
        img_a, img_b, metadata = self.dataset.get_pair(i)

        # aflow contains pixel coordinates indicating where each
        # pixel from the left image ended up in the right image
        # as (x,y) pairs, but its shape is (H,W,2)
        img_a = self.norm(np.array(img_a))
        img_b = self.norm(np.array(img_b))
        aflow = np.float32(metadata['aflow'])
        mask = metadata.get('mask', np.ones(aflow.shape[:2], np.uint8))
        org_flow = np.float32(metadata['org_flow'])
        # img_a_patch = patch_extractor(torch.unsqueeze(torch.tensor(img_a), 0))
        # img_b_patch = patch_extractor(torch.unsqueeze(torch.tensor(img_b), 0))

        result = dict(
            img1=img_a,
            img2=img_b,
            aflow=aflow,
            mask=mask,
            org_flow=org_flow,
            # img1_patch=img_a_patch,
            # img2_patch=img_b_patch
        )
        return result


class SingleLoader:
    def __init__(self, dataset, crop, scale, distort):
        assert hasattr(dataset, 'npairs')
        assert hasattr(dataset, 'get_item')
        self.dataset = dataset
        self.crop = crop
        self.scale = scale
        self.distort = distort
        self.n_samples = 5  # number of random trials per image
        self.norm = normalize_img

    def __len__(self):
        assert len(self.dataset) == self.dataset.npairs, "not same length"  # and not nimg
        return len(self.dataset)

    def __getitem__(self, i):
        # Retrieve an image pair and their absolute flow
        img = self.dataset.get_item(i)
        return {'img': self.norm(np.array(img))}


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
        pin_memory=iscuda,
        collate_fn=collate)


def collate(batch, _use_shared_memory=True):
    """Puts each data field into a tensor with outer dimension batch size.
    Copied from https://github.com/pytorch in torch/utils/data/_utils/collate.py
    """
    import re
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        assert elem_type.__name__ == 'ndarray'
        # array of string classes and object
        if re.search('[SaUO]', elem.dtype.str) is not None:
            raise TypeError(error_msg.format(elem.dtype))
        batch = [torch.from_numpy(b) for b in batch]
        try:
            return torch.stack(batch, 0)
        except RuntimeError:
            return batch
    elif batch[0] is None:
        return list(batch)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], dict):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class SythPairLoader:
    """ On-the-fly jittering of pairs of image with dense pixel ground-truth correspondences.

        crop:   random crop applied to both images
        scale:  random scaling applied to img2
        distort: random ditorsion applied to img2

        self[idx] returns a dictionary with keys: img1, img2, aflow, mask
         - img1: cropped original
         - img2: distorted cropped original
         - aflow: 'absolute' optical flow = (x,y) position of each pixel from img1 in img2
         - mask: (binary image) valid pixels of img1
        """

    def __init__(self, crop, scale, distort,
                 what='aflow mask', idx_as_rng_seed=False):
        self.distort = distort
        self.crop = crop
        self.scale = scale
        self.idx_as_rng_seed = idx_as_rng_seed  # to remove randomness
        self.what = what.split() if isinstance(what, str) else what
        self.n_samples = 5  # number of random trials per image
        self.norm = normalize_img

    def __len__(self):
        return 1

    def getitem(self, img_a, img_b, metadata):
        aflow = np.float32(metadata['aflow'])
        # mask = metadata.get('mask', np.ones(aflow.shape[:2], np.uint8))
        mask = Image.fromarray(metadata.get('mask', np.ones(aflow.shape[:2], np.uint8)))
        mask = mask.resize(img_a.size, Image.BILINEAR)
        mask = np.asarray(mask)

        # apply transformations to the second image
        img_b = {'img': img_b, 'persp': (1, 0, 0, 0, 1, 0, 0, 0)}
        if self.scale:
            # print("dataloader img b size: ", np.array(img_b["img"]).shape)
            img_b = self.scale(img_b)
        if self.distort:
            img_b = self.distort(img_b)

            # apply the same transformation to the flow
        aflow[:] = persp_apply(img_b['persp'], aflow.reshape(-1, 2)).reshape(aflow.shape)
        corres = None
        if 'corres' in metadata:
            corres = np.float32(metadata['corres'])
            corres[:, 1] = persp_apply(img_b['persp'], corres[:, 1])

        # apply the same transformation to the homography
        homography = None
        if 'homography' in metadata:
            homography = np.float32(metadata['homography'])
            # p_b = homography * p_a
            persp = np.float32(img_b['persp'] + (1,)).reshape(3, 3)
            homography = persp @ homography

        # determine crop size
        img_b = img_b['img']
        crop_size = self.crop({'imsize': (10000, 10000)})['imsize']
        # output_size_a = min(img_a.size, crop_size)
        # output_size_b = min(img_b.size, crop_size)
        output_size_a = crop_size
        output_size_b = crop_size
        img_a = np.array(img_a)
        img_b = np.array(img_b)

        ah, aw = img_a.shape
        bh, bw = img_b.shape
        assert aflow.shape == (ah, aw, 2)
        assert mask.shape == (ah, aw)

        # Let's start by computing the scale of the
        # optical flow and applying a median filter:
        dx = np.gradient(aflow[:, :, 0])
        dy = np.gradient(aflow[:, :, 1])
        scale = np.sqrt(np.clip(np.abs(dx[1] * dy[0] - dx[0] * dy[1]), 1e-16, 1e16))

        accu2 = np.zeros((16, 16), bool)
        Q = lambda x, w: np.int32(16 * (x - w.start) / (w.stop - w.start))

        n_valid_pixel = mask.sum()
        sample_w = mask / (1e-16 + n_valid_pixel)

        # Find suitable left and right windows
        trials = 0  # take the best out of few trials
        best = -np.inf, None
        best_windows = []

        def sample_valid_pixel():
            n = np.random.choice(sample_w.size, p=sample_w.ravel())
            y, x = np.unravel_index(n, sample_w.shape)
            return x, y

        def window1(x, size, w):
            l = x - int(0.5 + size / 2)
            r = l + int(0.5 + size)
            if l < 0: l, r = (0, r - l)
            if r > w: l, r = (l + w - r, w)
            if l < 0: l, r = 0, w  # larger than width
            return slice(l, r)

        def window(cx, cy, win_size, scale, img_shape):
            return (window1(cy, win_size[1] * scale, img_shape[0]),
                    window1(cx, win_size[0] * scale, img_shape[1]))

        for _ in range(50 * self.n_samples):
            # finished!
            if trials >= self.n_samples:
                break
            # pick a random valid point from the first image
            if n_valid_pixel == 0:
                break

            c1x, c1y = sample_valid_pixel()
            # Find in which position the center of the left
            # window ended up being placed in the right image
            c2x, c2y = (aflow[c1y, c1x] + 0.5).astype(np.int32)
            if not (0 <= c2x < bw and 0 <= c2y < bh):
                continue

                # Get the flow scale
            sigma = scale[c1y, c1x]
            print(sigma)

            # Determine sampling windows
            if 0.2 < sigma < 1:
                win1 = window(c1x, c1y, output_size_a, 1 / sigma, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, 1, img_b.shape)
            elif 1 <= sigma < 5:
                win1 = window(c1x, c1y, output_size_a, 1, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, sigma, img_b.shape)
            else:
                continue  # bad scale

            # compute a score based on the flow
            x2, y2 = aflow[win1].reshape(-1, 2).T.astype(np.int32)

            # Check the proportion of valid flow vectors
            valid = (win2[1].start <= x2) & (x2 < win2[1].stop) & (win2[0].start <= y2) & (y2 < win2[0].stop)
            score1 = (valid * mask[win1].ravel()).mean()

            # check the coverage of the second window
            accu2[:] = False
            accu2[Q(y2[valid], win2[0]), Q(x2[valid], win2[1])] = True
            score2 = accu2.mean()

            # Check how many hits we got
            score = min(score1, score2)

            trials += 1
            if score > best[0]:
                best = score, win1, win2
                best_windows.append(best)

            print(best)

        if None in best:  # counldn't find a good window
            img_a = np.zeros(output_size_a[::-1] + (3,), dtype=np.uint8)
            img_b = np.zeros(output_size_b[::-1] + (3,), dtype=np.uint8)
            aflow = np.nan * np.ones((2,) + output_size_a[::-1], dtype=np.float32)
            homography = np.nan * np.ones((3, 3), dtype=np.float32)

        else:
            win1, win2 = best[1:]
            img_a = img_a[win1]
            img_b = img_b[win2]
            aflow = aflow[win1] - np.float32([[[win2[1].start, win2[0].start]]])
            mask = mask[win1]
            aflow[~mask] = np.nan  # mask bad pixels!
            aflow = aflow.transpose(2, 0, 1)  # --> (2,H,W)

            if corres is not None:
                corres[:, 0] -= (win1[1].start, win1[0].start)
                corres[:, 1] -= (win2[1].start, win2[0].start)

            if homography is not None:
                trans1 = np.eye(3, dtype=np.float32)
                trans1[:2, 2] = (win1[1].start, win1[0].start)
                trans2 = np.eye(3, dtype=np.float32)
                trans2[:2, 2] = (-win2[1].start, -win2[0].start)
                homography = trans2 @ homography @ trans1
                homography /= homography[2, 2]

                # rescale if necessary
            if img_a.shape[:2][::-1] != output_size_a:
                sx, sy = (np.float32(output_size_a) - 1) / (np.float32(img_a.shape[:2][::-1]) - 1)
                img_a = np.asarray(Image.fromarray(img_a).resize(output_size_a, Image.ANTIALIAS))
                mask = np.asarray(Image.fromarray(mask).resize(output_size_a, Image.NEAREST))
                afx = Image.fromarray(aflow[0]).resize(output_size_a, Image.NEAREST)
                afy = Image.fromarray(aflow[1]).resize(output_size_a, Image.NEAREST)
                aflow = np.stack((np.float32(afx), np.float32(afy)))

                if corres is not None:
                    corres[:, 0] *= (sx, sy)

                if homography is not None:
                    homography = homography @ np.diag(np.float32([1 / sx, 1 / sy, 1]))
                    homography /= homography[2, 2]

            if img_b.shape[:2][::-1] != output_size_b:
                sx, sy = (np.float32(output_size_b) - 1) / (np.float32(img_b.shape[:2][::-1]) - 1)
                img_b = np.asarray(Image.fromarray(img_b).resize(output_size_b, Image.ANTIALIAS))
                aflow *= [[[sx]], [[sy]]]

                if corres is not None:
                    corres[:, 1] *= (sx, sy)

                if homography is not None:
                    homography = np.diag(np.float32([sx, sy, 1])) @ homography
                    homography /= homography[2, 2]

        assert aflow.dtype == np.float32, pdb.set_trace()
        assert homography is None or homography.dtype == np.float32, pdb.set_trace()
        assert img_b.shape == img_a.shape
        if 'flow' in self.what:
            H, W = img_a.shape[:2]
            mgrid = np.mgrid[0:H, 0:W][::-1].astype(np.float32)
            # flow = aflow - mgrid
            flow = aflow

        result = dict(img1=self.norm(img_a), img2=self.norm(img_b))
        for what in self.what:
            try:
                result[what] = eval(what)
            except NameError:
                pass
        return result


