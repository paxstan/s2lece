import extract
from input_pipeline.dataset import SyntheticPairDataset
import numpy as np
import cv2
from utils.data_conversion import project_point_cloud
import matplotlib.pyplot as plt
import open3d as o3d
import os


# estimate transformation based on key points and descriptors of two point clouds with RANSAC
def execute_global_registration(source_down, target_down, reference_desc, target_desc, distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down, target=target_down, source_feature=reference_desc, target_feature=target_desc,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 600), mutual_filter=True)
    return result


# remove invalid key points from extracted xys, scores and desc
def remove_invalid_keypoints(xys, scores, desc, mask):
    valid_mask = mask[xys[:, 1].astype(int), xys[:, 0].astype(int)]
    xys = xys[valid_mask]
    scores = scores[valid_mask]
    desc = desc[valid_mask]
    return xys, scores, desc


# draw two misaligned point clouds after aligning them with transformation
def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.8, 0.4, 0])
    target_temp.paint_uniform_color([0, 0.6, 0.6])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def match_fun(config):
    net = extract.load_network(config["save_path"])
    net = net.cuda()
    show = True

    lidar_pair_dt = SyntheticPairDataset(root=config["data_dir"])
    idx1 = "0"
    imgA = lidar_pair_dt.get_image(idx1)
    maskA = lidar_pair_dt.get_valid_range_mask(idx1)
    xysA, scoresA, descA = extract.extract_keypoints(imgA, config, net)
    xysA, scoresA, descA = remove_invalid_keypoints(xysA, scoresA, descA, maskA)
    xyzA = lidar_pair_dt.get_xyz(idx1)
    xyzA_sort = xyzA[xysA[:, 1].astype(int), xysA[:, 0].astype(int)]
    xyzA = xyzA.reshape((-1, 3))[maskA.reshape(-1)]

    idx2 = "5"
    imgB = lidar_pair_dt.get_image(idx2)
    maskB = lidar_pair_dt.get_valid_range_mask(idx2)
    xysB, scoresB, descB = extract.extract_keypoints(imgB, config, net)
    xysB, scoresB, descB = remove_invalid_keypoints(xysB, scoresB, descB, maskB)
    xyzB = lidar_pair_dt.get_xyz(idx2)
    xyzB_sort = xyzB[xysB[:, 1].astype(int), xysB[:, 0].astype(int)]
    xyzB = xyzB.reshape((-1, 3))[maskB.reshape(-1)]

    # cast data to open3d format
    reference_pc = o3d.geometry.PointCloud()
    reference_pc.points = o3d.utility.Vector3dVector(xyzA)
    ref = o3d.pipelines.registration.Feature()
    ref.data = descA.T
    ref_key = o3d.geometry.PointCloud()
    ref_key.points = o3d.utility.Vector3dVector(xyzA_sort)

    test_pc = o3d.geometry.PointCloud()
    test_pc.points = o3d.utility.Vector3dVector(xyzB)
    test = o3d.pipelines.registration.Feature()
    test.data = descB.T
    test_key = o3d.geometry.PointCloud()
    test_key.points = o3d.utility.Vector3dVector(xyzB_sort)

    # get tansformation estimate and set of correct matches with RANSAC
    result_ransac = execute_global_registration(ref_key, test_key, ref, test, 0.75)
    tf = result_ransac.transformation
    matches = np.array(result_ransac.correspondence_set)

    # show results
    if show:
        # Plot point clouds after registration
        reference_pc.paint_uniform_color([0.8, 0.4, 0])
        test_pc.paint_uniform_color([0, 0.6, 0.6])
        match_lines = o3d.geometry.LineSet()
        match_lines.points = o3d.utility.Vector3dVector(np.concatenate((xyzA_sort, xyzB_sort), axis=0))
        # matches[:, 1] = matches[:, 1] + xyzA_sort.shape[0]  # select matches of second image
        match_lines.lines = o3d.utility.Vector2iVector(matches)
        match_lines.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([reference_pc, test_pc, match_lines])
        # draw_registration_result(reference_pc, test_pc, tf)

        def draw_circles(xys, img, matches):
            x = xys[matches, 0].astype(int)
            y = xys[matches, 1].astype(int)
            # r, i, s = img.split()
            i = np.array(img)
            # i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
            # i = np.array(img)
            for k in range(x.shape[0]):
                # if not mask[k]: continue
                i = cv2.circle(i, (x[k], y[k]), 2, (0, 0, 255), 1)
            return i

        # draw key point circles
        int_w_circ_A = draw_circles(xysA, imgA, matches[:, 0].astype(int))
        int_w_circ_B = draw_circles(xysB, imgB, matches[:, 1].astype(int))
        stacked = np.vstack((int_w_circ_A, int_w_circ_B))

        # draw match lines
        thickness = 1
        lineType = cv2.LINE_AA
        h = imgA.size[1]
        for j in range(matches.shape[0]):
            x1 = xysA[matches[:, 0].astype(int), 0][j]
            y1 = xysA[matches[:, 0].astype(int), 1][j]
            x2 = xysB[matches[:, 1].astype(int), 0][j]
            y2 = xysB[matches[:, 1].astype(int), 1][j] + h
            color = (0, 255, 0)
            cv2.line(stacked, (x1, y1), (x2, y2), color, thickness, lineType)

        win_inp = 'Keypoints'
        cv2.namedWindow(win_inp)
        cv2.imshow(win_inp, stacked)
        plt.show()
        cv2.waitKey(0)
