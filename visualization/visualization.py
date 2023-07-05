# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from models.utils import coords_grid


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Copied from https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
    Copyright (c) 2018 Tom Runia
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:

    Copied from https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
    Copyright (c) 2018 Tom Runia
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:

    Copied from https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
    Copyright (c) 2018 Tom Runia
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


def show_flow(img0, img1, flow, mask=None):
    img0 = np.asarray(img0)
    img1 = np.asarray(img1)
    if mask is None: mask = 1
    mask = np.asarray(mask)
    if mask.ndim == 2: mask = mask[:, :, None]
    assert flow.ndim == 3
    assert flow.shape[:2] == img0.shape[:2] and flow.shape[2] == 2

    def noticks():
        plt.xticks([])
        plt.yticks([])

    fig = plt.figure("showing correspondences")
    ax1 = plt.subplot(221)
    ax1.numaxis = 0
    plt.imshow(img0 * mask)
    noticks()
    ax2 = plt.subplot(222)
    ax2.numaxis = 1
    plt.imshow(img1)
    noticks()

    ax = plt.subplot(212)
    ax.numaxis = 0
    flow_img = flow_to_color(np.where(np.isnan(flow), 0, flow))
    plt.imshow(flow_img * mask)
    noticks()

    plt.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.02, hspace=0.02)

    def motion_notify_callback(event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        # remove all lines
        while ax1.lines:
            ax1.lines[0].remove()
        while ax2.lines:
            ax2.lines[0].remove()
        # ax1.lines = []
        # ax2.lines = []
        try:
            x, y = int(x + 0.5), int(y + 0.5)
            ax1.plot(x, y, '+', ms=10, mew=2, color='blue', scalex=False, scaley=False)
            x, y = flow[y, x] + (x, y)
            ax2.plot(x, y, '+', ms=10, mew=2, color='red', scalex=False, scaley=False)
            # we redraw only the concerned axes
            renderer = fig.canvas.get_renderer()
            ax1.draw(renderer)
            ax2.draw(renderer)
            fig.canvas.blit(ax1.bbox)
            fig.canvas.blit(ax2.bbox)
        except IndexError:
            return

    cid_move = fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)
    print("Move your mouse over the images to show matches (ctrl-C to quit)")
    plt.show(block=True)


def flow2rgb(flow_map, max_value=None):
    flow_map_np = np.floor(flow_map.detach().squeeze().numpy())
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.nanmax(np.abs(flow_map_np)))
    rgb_map[0] += normalized_flow_map[0]  # more red if flow[0] is large
    rgb_map[1] -= 0.5 * (
            normalized_flow_map[0] + normalized_flow_map[1])  # more blue if in between displacement is large
    rgb_map[2] += normalized_flow_map[1]  # more green if flow[1] is large
    return rgb_map.clip(0, 1)


def display_flows(**kwargs):
    # Create subplots
    fig, axes = plt.subplots(len(kwargs), 1, sharex=True, sharey=True)

    for index, (key, value) in enumerate(kwargs.items()):
        rgb_flow = flow2rgb(20 * value, max_value=None)
        img = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
        axes[index].imshow(img, cmap='gray')
        axes[index].set_title(key)
    plt.tight_layout()
    plt.show()


def visualize_correlation(corr_result, grid_size):
    corr_result = corr_result.detach().squeeze().numpy()
    fig, axs = plt.subplots(grid_size, grid_size, sharex=True, sharey=True)
    for i in range(grid_size):
        for j in range(grid_size):
            # Select the channel to plot
            channel = corr_result[i * grid_size + j]

            # Plot the channel in the corresponding subplot
            axs[i, j].imshow(channel)
            axs[i, j].set_title(f'Channel {i * grid_size + j}')
            axs[i, j].axis('off')


def compare_flow(target_flow, pred_flow, valid_masks, idx=1, loss=0):
    pred_last_np = np.floor(pred_flow[-1][0, :, :, :]
                            .cpu().detach().numpy()).transpose(1, 2, 0).reshape(32 * 1024, 2)
    invalid_mask = ~valid_masks[-1][0, :, :].cpu().detach().numpy().flatten()
    pred_last_np[invalid_mask, :] = 0

    pred_flow_img = flow_to_color(pred_last_np.reshape(32, 1024, 2))
    true_flow_img = flow_to_color(target_flow[0, :, :, :].cpu().detach().squeeze().numpy().transpose(1, 2, 0))

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    axes[0].imshow(true_flow_img.squeeze())
    axes[0].set_title("original flow")

    axes[1].imshow(pred_flow_img.squeeze())
    axes[1].set_title("predicted flow")

    # plt.text(2.5, -5, f'loss: {loss}', ha='center')
    plt.figtext(0.5, 0.05, f'loss: {loss}', ha='center')

    plt.savefig(f"runs/pred_optical_flow_{idx}.png")
    # plt.show()


def visualize_point_cloud(pred_flow, metadata, transform=False):
    pred_flow = pred_flow.detach().squeeze().numpy()
    c, h, w = pred_flow.shape
    abs_flow = np.zeros_like(pred_flow)
    abs_flow[0, :, :] = np.arange(h)[:, np.newaxis]
    abs_flow[1, :, :] = np.arange(w)
    abs_flow = np.floor(abs_flow + pred_flow)
    abs_flow = abs_flow.transpose(1, 2, 0)
    x_img = abs_flow[:, :, 0].reshape(-1).astype(int)
    y_img = abs_flow[:, :, 1].reshape(-1).astype(int)
    mask_valid = (x_img >= 0) * (x_img < h) * (y_img >= 0) * (y_img < w)
    x_img[np.invert(mask_valid)] = 0
    y_img[np.invert(mask_valid)] = 0
    idx1 = metadata['idx1'].reshape(-1, 1)
    idx2 = metadata['idx2']
    corres_idx2 = (idx2[x_img.astype(int), y_img.astype(int)]).reshape(-1, 1)
    corres_id = np.hstack((idx1, corres_idx2))
    corres_id[np.invert(mask_valid), :] = [-1, -1]
    valid_corres_id = corres_id[~np.all(corres_id == [-1, -1], axis=1)]
    # valid_corres_id = valid_corres_id[~np.all(valid_corres_id[:, 1] == 0)]

    # Create two point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(metadata['xyz1'])  # Random point cloud 1
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(metadata['xyz2'])  # Random point cloud 2

    # Assign colors to each point cloud
    color1 = [1.0, 0.0, 0.0]  # Red color for point cloud 1
    color2 = [0.0, 1.0, 0.0]  # Green color for point cloud 2

    pcd1.paint_uniform_color(color1)  # Assign color1 to all points in pcd1
    pcd2.paint_uniform_color(color2)  # Assign color2 to all points in pcd2

    if transform:
        # Estimate transformation using correspondences
        transformation = o3d.pipelines.registration.TransformationEstimationPointToPoint() \
            .compute_transformation(pcd2, pcd1, o3d.utility.Vector2iVector(valid_corres_id))

        # Apply transformation to align the second point cloud to the first
        pcd2.transform(transformation)

    # Create visualization options
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point clouds to the visualizer
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    # Create lines between corresponding points
    lines = []
    for i in range(valid_corres_id.shape[0] // 10):
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(
            [pcd1.points[valid_corres_id[i][0]], pcd2.points[valid_corres_id[i][1]]])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        lines.append(line)
        vis.add_geometry(line)

    # Visualize the point clouds and lines
    vis.run()

    # Capture a screenshot of the visualizer window
    vis.capture_screen_image(filename="runs/visualization.png")

    # Save the image
    # o3d.io.write_image("visualization.png", image)

    vis.destroy_window()
