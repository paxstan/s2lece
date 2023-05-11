import cv2
import numpy as np
from PIL import Image
from input_pipeline.dataset import SyntheticPairDataset
import open3d as o3d
from utils.extract_keypoints import *
from models.patchnet import *


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


