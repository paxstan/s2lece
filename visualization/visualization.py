import logging
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
matplotlib.use('Agg')


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

    # rad = np.sqrt(np.square(u) + np.square(v))
    # rad_max = np.max(rad)

    epsilon = 1e-5
    # u = u / (rad_max + epsilon)
    # v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


def compare_flow(target_flow, pred_flow, path, idx=1, loss=0):
    """
    function to visualize flow comparison
    :param target_flow: ground truth optical flow
    :param pred_flow: predicted optical flow
    :param path: path to save
    :param idx: id of the comparison
    :param loss: loss between target and predicted
    """
    predicted_flow = np.floor(pred_flow[0, :, :, :].cpu().detach().numpy()).transpose(1, 2, 0)
    target_flow = target_flow[0, :, :, :].cpu().detach().squeeze().numpy().transpose(1, 2, 0)

    pred_flow_img = flow_to_color(predicted_flow)
    true_flow_img = flow_to_color(target_flow)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    axes[0].imshow(true_flow_img)
    axes[0].set_title("original flow")

    axes[1].imshow(pred_flow_img)
    axes[1].set_title("predicted flow")

    # plt.text(2.5, -5, f'loss: {loss}', ha='center')
    plt.figtext(0.5, 0.05, f'loss: {loss}', ha='center')

    plt.savefig(f"{path}/{idx}_pred_optical_flow.png", dpi=300)

    plt.close(fig)

    np.save(f"{path}/{idx}_pred_optical_flow.npy", predicted_flow)
    np.save(f"{path}/{idx}_target_optical_flow.npy", target_flow)


def show_visual_progress(org_img, pred_img, path, title=None, loss=0):
    """
    function to visualize comparison between actual and predicted image from autoencoder
    :param org_img: actual image
    :param pred_img: predicted image
    :param path: path to save the comparison
    :param title: name of the comparison image file
    :param loss: loss between actual and predicted image
    """
    try:
        org_img = org_img.detach().cpu().numpy()[0, 0]
        pred_img = pred_img.detach().cpu().numpy()[0, 0]

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        axes[0].imshow(org_img)
        axes[0].set_title("original image")
        axes[1].imshow(pred_img)
        axes[0].set_title("predicted image")

        plt.figtext(0.5, 0.05, f'loss: {loss}', ha='center')

        if title:
            title = title.replace(" ", "_")
            plt.savefig(f"{path}/{title}", dpi=300)
        plt.close(fig)
    except Exception as e:
        logging.info(f"Issue in show  progress function : Exception: {e}")


def visualize_point_cloud(pred_flow, initial_flow, idx1, idx2, xyz1, xyz2, mask_valid, path, transform=False):
    """
    function to extract correspondence between source and target from optical flow
    """
    pred_flow = pred_flow.detach().squeeze().numpy()
    initial_flow = initial_flow.detach().squeeze().numpy()
    idx1 = idx1.detach().squeeze().numpy()
    idx2 = idx2.detach().squeeze().numpy()
    xyz1 = xyz1.detach().squeeze().numpy()
    xyz2 = xyz2.detach().squeeze().numpy()
    mask_valid = mask_valid.detach().squeeze().numpy()

    abs_flow = np.floor(initial_flow + pred_flow)
    x_img = abs_flow[0].astype(int)
    y_img = abs_flow[1].astype(int)

    inv_x = np.where(x_img >= 64)
    inv_y = np.where(y_img >= 2048)

    x_img[inv_x] = inv_x[0]
    y_img[inv_y] = inv_y[1]

    x_img = x_img.reshape(-1)
    y_img = y_img.reshape(-1)

    corres_idx2 = (idx2[x_img.astype(int), y_img.astype(int)])
    count = np.bincount(corres_idx2[corres_idx2 != -1])
    non_unique_val = np.where(count > 1)[0]
    non_unique_indices = np.array([])
    for i in non_unique_val:
        non_unique_id = np.where(corres_idx2 == i)[0]
        non_unique_indices = np.concatenate((non_unique_indices, non_unique_id))
    non_unique_indices = non_unique_indices.astype(int)
    corres_idx2[non_unique_indices] = -1

    corres_idx2 = corres_idx2.reshape(-1, 1)

    corres_id = np.hstack((idx1.reshape(-1, 1), corres_idx2))
    corres_id[np.invert(mask_valid.flatten()), :] = [-1, -1]

    valid_index = np.where((corres_id[:, 0] != -1) & (corres_id[:, 1] != -1))[0]
    valid_corres_id = corres_id[valid_index]

    visualize_correspondence(xyz1, xyz2, valid_corres_id, transform, path)


def visualize_correspondence(source_point, target_point, valid_corres_id, transform, path):
    """function to visualize correspondence"""
    # Create two point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source_point)  # Random point cloud 1
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target_point)  # Random point cloud 2

    # Assign colors to each point cloud
    color1 = [1.0, 0.0, 0.0]  # Red color for point cloud 1
    color2 = [0.0, 1.0, 0.0]  # Green color for point cloud 2

    pcd1.paint_uniform_color(color1)  # Assign color1 to all points in pcd1
    pcd2.paint_uniform_color(color2)  # Assign color2 to all points in pcd2

    if transform:
        # Estimate transformation using correspondences
        transformation = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(
            pcd2, pcd1, o3d.utility.Vector2iVector(valid_corres_id[:, [1, 0]]))

        print("Apply point-to-point ICP")
        reg_p2l = o3d.pipelines.registration.registration_icp(
            pcd2, pcd1, 0.001, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        # print(reg_p2l)
        # print("Transformation is:")
        # print(reg_p2l.transformation)

        # Apply transformation to align the second point cloud to the first
        pcd2.transform(reg_p2l.transformation)

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
    vis.capture_screen_image(filename=f"{path}_vis.png")

    vis.destroy_window()


def visualize_different_viewpoints(point_cloud):
    # Load your point cloud data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Number of viewpoints
    num_viewpoints = 5

    # Create a visualization window
    vis = o3d.visualization.Visualizer()

    for i in range(num_viewpoints):
        # Generate random rotation angles around X, Y, and Z axes
        rand_rot_x = np.random.uniform(0, 2 * np.pi)
        rand_rot_y = np.random.uniform(0, 2 * np.pi)
        rand_rot_z = np.random.uniform(0, 2 * np.pi)

        # Generate random translation along X, Y, and Z axes
        rand_trans_x = np.random.uniform(-1, 1)
        rand_trans_y = np.random.uniform(-1, 1)
        rand_trans_z = np.random.uniform(-1, 1)

        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Create translation matrix
        translation_matrix = np.array([
            [1, 0, 0, rand_trans_x],
            [0, 1, 0, 0],
            [0, 0, 1, rand_trans_z],
            [0, 0, 0, 1]
        ])

        # Combine rotation and translation to get the transformation matrix
        transformation_matrix = translation_matrix @ rotation_matrix

        # Deep copy the original point cloud before applying transformations
        transformed_cloud = copy.deepcopy(pcd)

        # Apply the transformation to the point cloud
        transformed_cloud.transform(transformation_matrix)

        o3d.io.write_point_cloud(f"../point_cloud_t_{i}.pcd", transformed_cloud)

    # Close the visualization window
    vis.destroy_window()
