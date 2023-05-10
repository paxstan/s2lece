import os, pdb
import cv2
import numpy as np
import torch
from PIL import Image
from utils import common
from input_pipeline.dataloader import normalize_img
from models.patchnet import *
from input_pipeline.dataset import SyntheticPairDataset
import time
import open3d as o3d


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, one, H, W = img.shape
    assert B == 1 and one == 1, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                # start = time.time()

                res = net(imgs=[img[:, :2, :, :]])
                # print(time.time() - start)

            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y, x = detector(**res)  # nms
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def extract_keypoints(img, config, net):
    iscuda = common.torch_set_gpu(config["gpu"])

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr=config["reliability_thr"],
        rep_thr=config["repeatability_thr"])

    W, H = img.size
    im = normalize_img(img)[None]
    if iscuda: im = im.cuda()

    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, im, detector,
                                           scale_f=2**0.25,
                                           min_scale=1,
                                           max_scale=1,
                                           min_size=1024,
                                           max_size=1024,
                                           verbose=False)

    xys = xys.cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = scores.argsort()[-1000 or None:]

    xys = xys[idxs, :].astype(int)
    scores = scores[idxs]
    desc = desc[idxs, :]

    # remove keypoints close to border
    border = 4
    mask = (xys[:, 0] >= border) * (xys[:, 0] < W - border) * (xys[:, 1] >= border) * (xys[:, 1] < H - border)
    xys = xys[mask, :]
    scores = scores[mask]
    desc = desc[mask, :]

    return xys, scores, desc


def test_model(config):
    lidar_pair_dt = SyntheticPairDataset(root=config["data_dir"])
    idx = "0"
    img = lidar_pair_dt.get_image(idx)
    mask = lidar_pair_dt.get_valid_range_mask(idx)
    net = load_network(config["save_path"])
    net = net.cuda()
    xys, scores, desc = extract_keypoints(img, config, net)
    show = True
    if show:
        def blended(xys, img, matches):
            x = xys[matches, 0].astype(int)
            y = xys[matches, 1].astype(int)
            # r, i, s = img.split()
            i = np.array(img)
            # i *= 5
            # i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
            for k in range(x.shape[0]):
                if mask[y[k], x[k]] and scores[k] > 0.0:
                    i = cv2.circle(i, (x[k], y[k]), 2, (0, 0, 255), 1)
            return i

        matches = np.ones_like(xys)[:, 0]
        blend = blended(xys, img, matches.astype(bool))
        Image.fromarray(blend).show()

        xyz = lidar_pair_dt.get_xyz(idx)
        keypoint_mask = np.ones_like(xyz[:, :, 0], dtype=bool)
        keypoint_mask[xys[:, 1], xys[:, 0]] = False

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz[keypoint_mask, :].reshape(-1, 3))
        i = np.array(img)
        print(i.shape)
        i = i[keypoint_mask].reshape(-1) / 255
        colors = [[i[k], i[k], i[k]] for k in range(i.shape[0])]
        pc.colors = o3d.utility.Vector3dVector(colors)

        xyz = xyz[xys[:, 1], xys[:, 0], :]

        keypoints = o3d.geometry.PointCloud()
        keypoints.points = o3d.utility.Vector3dVector(xyz)

        colors = [[0.5 * (1 + scores[i]), 0, 0] for i in range(xyz.shape[0])]
        keypoints.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pc, keypoints])


