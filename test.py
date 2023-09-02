import os
import copy
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import cv2
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from utils.data_conversion import (project_point_cloud, perform_kdtree, LidarDataConverter, build_flow, map_points_xy,
                                   get_pixel_match)
from visualization.visualization import visualize_correspondence, flow_to_color, visualize_point_cloud, \
    visualize_different_viewpoints
from scipy.spatial import cKDTree
import pandas as pd
import utils.data_conversion


def projection_convert(points, height=32, width=1024):
    proj_range = np.full((height, width), -1, dtype=np.float32)

    # unprojected range (list of depths for each point)
    un_proj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((height, width, 3), -1, dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((height, width), -1, dtype=np.int32)
    proj_x, proj_y, depth = project_point_cloud(points, height=height, width=width)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    # assigning to images
    proj_range[proj_x, proj_y] = depth
    proj_xyz[proj_x, proj_y] = points
    proj_idx[proj_x, proj_y] = indices
    proj_mask = (proj_idx > 0).astype(np.int32)

    return proj_xyz, proj_range, proj_mask, proj_idx


def perform_icp(source, target):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target)
    # Perform ICP registration
    threshold = 0.02  # Maximum correspondence distance
    trans_init = np.identity(4)  # Initial transformation (identity matrix)
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint())

    correspondence = np.asarray(reg_p2p.correspondence_set)
    # visualize_correspondence(source, target, correspondence, False)
    # print(correspondence.shape)
    return correspondence, reg_p2p.fitness


def read_bag():
    i = 0
    pcd_set = []
    list_gt = []
    gt_gen_path = "../dataset/exp05/ground_truth_pose.npy"
    bag = rosbag.Bag("../../data/hilti/rosbag/exp05_imu/exp05_construction_upper_level_2.bag")
    gt_file_path = "../../data/hilti/rosbag/exp05_imu/exp_05_construction_upper_level_2_imu.txt"

    with open(gt_file_path) as file:
        ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])
    ground_truth_imu = ground_truth_imu[ground_truth_imu[:, 0].argsort()]

    try:
        for topic, msg, time in bag.read_messages(topics=['/hesai/pandar']):
            assert i < len(ground_truth_imu) - 1, "i should be less than length of gt file"
            if ground_truth_imu[i][0] == time.to_time():
                data = list(pc2.read_points(msg, skip_nans=True,
                                            field_names=['x', 'y', 'z', 'timestamp']))
                cloud_points, timestamp_data = np.hsplit(np.array(data), [3])
                visualize_different_viewpoints(cloud_points)
                # pcd_set.append({'id': i, 'data': data, 'imu': ground_truth_imu[i]})
                # list_gt.append(ground_truth_imu[i])
                # print(f"processed : {i}")
                # i = i + 1
            # if len(pcd_set) == 2:
            #     perform_icp(pcd_set[0], pcd_set[1])
            #     pcd_set = []
            # range_img, valid_mask = projection_convert(cloud_points)
            # range_img_new, valid_mask_new = projection_convert(cloud_points, width=2000)
            # range_img[np.invert(valid_mask)] = 0
            # range_img_new[np.invert(valid_mask_new)] = 0
            #
            # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
            # axs[0].imshow(range_img, cmap="gray")
            # axs[1].imshow(range_img_new, cmap="gray")
            # plt.show()


    except AssertionError as error:
        print(error)

    bag.close()
    np.save(gt_gen_path, np.array(list_gt))
    bag.close()

    return pcd_set


def perform_icp_on_set(ground_truth_imu, pcd_set):
    distance = 0
    j = 0
    for i in range(len(ground_truth_imu)):
        assert i < 200, "i should be less than length of gt file"
        curr_gt = ground_truth_imu[j]
        post_gt = ground_truth_imu[i]
        t_dist = np.sqrt(
            (post_gt[1] - curr_gt[1]) ** 2 + (post_gt[2] - curr_gt[2]) ** 2 + (post_gt[3] - curr_gt[3]) ** 2)
        if t_dist >= 2:
            print(f"source: {j}, dest: {i}, distance: {t_dist}")
            perform_icp(pcd_set[j], pcd_set[i])
            distance = distance + t_dist
            j = i


def find_u_v(idx, indices):
    id_dict = []
    for i in indices:
        index = np.where(idx == i)
        id_dict.append(list(zip(index[0], index[1])))
    return id_dict


def map_domain(s_idx, t_idx, indices, n_mask):
    # sort_unique_id = np.sort(np.unique(indices))
    # id_s_idx = np.where(np.isin(s_idx, np.arange(len(indices))))
    # id_t_idx = np.where(np.isin(t_idx, sort_unique_id))

    id_s = [np.column_stack(np.where(s_idx == i)).ravel().tolist() if i in s_idx else [np.nan, np.nan] for i in
            range(len(indices))]
    id_t = [np.column_stack(np.where(t_idx == i)).ravel().tolist() if i in t_idx else [np.nan, np.nan] for i in indices]

    id_s = np.array(id_s)
    id_t = np.array(id_t)

    diff_x = id_t[:, 0] - id_s[:, 0]
    diff_y = id_t[:, 1] - id_s[:, 1]

    mask = (np.isnan(diff_x)) | (np.isnan(diff_y)) | np.invert(n_mask)

    id_s = id_s[np.invert(mask)].astype(int)
    diff_x = diff_x[np.invert(mask)].astype(int)
    diff_y = diff_y[np.invert(mask)].astype(int)

    flow = np.zeros((2, 32, 2000))
    s_flow = np.zeros((2, 32, 2000))
    x = id_s[:, 0]
    y = id_s[:, 1]
    flow[0, x, y] = diff_x
    flow[1, x, y] = diff_y

    s_flow[0, x, y] = x
    s_flow[1, x, y] = y

    return flow, s_flow


# def build_flow(source_idx, target_idx, indices, n_mask):
#     s_idx = source_idx
#     t_idx = target_idx[indices]
#
#     diff_x = t_idx[:, 0] - s_idx[:, 0]
#     diff_y = t_idx[:, 1] - s_idx[:, 1]
#
#     mask = (np.isnan(diff_x)) | (np.isnan(diff_y)) | np.invert(n_mask)
#
#     id_s = s_idx[np.invert(mask)].astype(int)
#     diff_x = diff_x[np.invert(mask)].astype(int)
#     diff_y = diff_y[np.invert(mask)].astype(int)
#
#     flow = np.zeros((2, 32, 2000))
#     s_flow = np.zeros((2, 32, 2000))
#     x = id_s[:, 0]
#     y = id_s[:, 1]
#     flow[0, x, y] = diff_x
#     flow[1, x, y] = diff_y
#
#     s_flow[0, x, y] = x
#     s_flow[1, x, y] = y
#
#     return flow, s_flow


def get_flow_matrix():
    source_idx = np.load(
        "/home/paxstan/Documents/research_project/code/s2lece/dataset/data_v6/0/idx.npy")
    target_idx = np.load(
        "/home/paxstan/Documents/research_project/code/s2lece/dataset/data_v6/19/idx.npy")

    source_points = np.load(
        "/home/paxstan/Documents/research_project/code/s2lece/dataset/data_v6/0/xyz.npy")
    target_points = np.load(
        "/home/paxstan/Documents/research_project/code/s2lece/dataset/data_v6/19/xyz.npy")

    w_source_points = np.load(
        "/home/paxstan/Documents/research_project/code/s2lece/dataset/data_v6/0/world_frame.npy")
    w_target_points = np.load(
        "/home/paxstan/Documents/research_project/code/s2lece/dataset/data_v6/19/world_frame.npy")

    # s_xyz, s_range_img, s_valid_mask, source_idx = projection_convert(source_points, width=2000)
    # t_xyz, t_range_img, t_valid_mask, target_idx = projection_convert(target_points, width=2000)

    source_u_v = np.full((w_source_points.shape[0], 2), np.nan)
    target_u_v = np.full((w_source_points.shape[0], 2), np.nan)

    source_p = np.zeros_like(w_source_points)
    target_p = np.zeros_like(w_target_points)

    mask_s_inv = source_idx != -1
    mask_t_inv = target_idx != -1

    s_valid_idx = np.argwhere(mask_s_inv)
    t_valid_idx = np.argwhere(mask_t_inv)

    s_valid_val = source_idx[s_valid_idx[:, 0], s_valid_idx[:, 1]]
    t_valid_val = target_idx[t_valid_idx[:, 0], t_valid_idx[:, 1]]

    # s_valid_p = source_points[s_valid_idx[:, 0], s_valid_idx[:, 1]]
    # t_valid_p = target_points[t_valid_idx[:, 0], t_valid_idx[:, 1]]

    s_sort_id = np.argsort(s_valid_val)
    t_sort_id = np.argsort(t_valid_val)

    source_u_v[s_valid_val[s_sort_id]] = s_valid_idx[s_sort_id]
    target_u_v[t_valid_val[t_sort_id]] = t_valid_idx[t_sort_id]

    # source_p[s_valid_val[s_sort_id]] = s_valid_p[s_sort_id]
    # target_p[t_valid_val[t_sort_id]] = t_valid_p[t_sort_id]

    source_p = source_points
    target_p = target_points

    # unique_elements, counts = np.unique(indices, return_counts=True)
    distances, indices = perform_kdtree(w_source_points, w_target_points)
    nearest_nn = np.zeros((len(indices), 2))
    nearest_nn[:, 0] = np.arange(len(indices))
    nearest_nn[:, 1] = indices
    nearest_nn = nearest_nn.astype(int)
    # unique, counts = np.unique(indices, return_counts=True)
    # non_unique = unique[counts > 1]
    # u_unique = unique[counts == 1]
    # ii = np.where(indices == non_unique[0])
    # min_distance_id = distances[ii].argmin()
    # print(ii[0][min_distance_id])

    count = np.bincount(indices)

    true_val = np.where(count == 1)[0]
    true_indices = [np.column_stack(np.where(indices == i)).ravel().tolist() for i in true_val]
    true_indices = np.array(true_indices)
    unique_mask = np.zeros_like(indices)
    unique_mask[true_indices] = 1

    non_unique = np.where(count > 1)[0]
    non_unique_pair = [[i, np.column_stack(np.where(indices == i)).ravel().tolist()] for i in non_unique]
    # non_unique_pair2 = non_unique_pair
    for i, val in enumerate(non_unique_pair):
        min_distance_id = distances[val[1]].argmin()
        unique_mask[val[1][min_distance_id]] = 1
        # non_unique_pair2[i] = [val[0], val[1][min_distance_id]]

    unique_mask = unique_mask.astype(bool)

    neighbour_mask = distances <= 0.05

    mask = neighbour_mask * unique_mask

    nearest_nn = nearest_nn[neighbour_mask]

    flow, s_flow = build_flow(source_u_v, target_u_v, indices, mask)

    flow_img = flow_to_color(flow.transpose(1, 2, 0))
    plt.imshow(flow_img)
    plt.show()

    metadata = {'idx1': source_idx, 'idx2': target_idx, 'xyz1': source_p, 'xyz2': target_p}

    # visualize_point_cloud(flow, s_valid_mask, metadata)

    abs_flow = np.floor(s_flow + flow)
    x_img = abs_flow[0].astype(int)
    y_img = abs_flow[1].astype(int)

    invalid_x = np.where(x_img >= 32)
    invalid_y = np.where(y_img >= 2000)

    x_img[invalid_x] = invalid_x[0]
    y_img[invalid_y] = invalid_y[1]

    x_img = x_img.reshape(-1)
    y_img = y_img.reshape(-1)
    idx1 = metadata['idx1'].reshape(-1, 1)
    idx2 = metadata['idx2']
    corres_idx2 = (idx2[x_img.astype(int), y_img.astype(int)]).reshape(-1, 1)

    corres_id = np.hstack((idx1, corres_idx2))

    valid_index = (corres_id[:, 0] != -1) & (corres_id[:, 1] != -1)
    valid_corres_id = corres_id[valid_index]

    # visualize_correspondence(metadata['xyz1'], metadata['xyz2'], valid_corres_id, True)
    visualize_correspondence(metadata['xyz1'], metadata['xyz2'], nearest_nn, True)
    # visualize_correspondence(w_source_points, w_target_points, nearest_nn, True)

    print("domains")


def view_best_distance_idx(idx):
    with open(
            "/home/paxstan/Documents/research_project/data/hilti/rosbag/exp05_imu"
            "/exp_05_construction_upper_level_2_imu.txt") as file:
        ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])

    poses = ground_truth_imu[:, 1:4]
    distances = np.linalg.norm((poses[idx] - poses), axis=1)
    index = np.where((distances >= 0.01) & (distances <= 1))
    valid_distance = distances[index]
    far_point_idx = index[0][valid_distance.argmax()]
    return far_point_idx
    # flow = np.load(f"../dataset/exp05/correspondence/corres_{idx}_{far_point_idx}/flow.npy")
    # flow_im = flow_to_color(flow.transpose(1, 2, 0))
    # plt.imshow(flow_im)


def draw_registration_result(source, target, target2):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_temp_2 = copy.deepcopy(target2)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    target_temp_2.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source_temp, target_temp, target_temp_2])


def synth_flow(source, target):
    source_xyz = np.load(os.path.join(source, 'xyz.npy'))
    target_xyz = np.load(os.path.join(target, 'xyz.npy'))

    source_idx = np.load(os.path.join(source, 'idx.npy'))
    target_idx = np.load(os.path.join(target, 'idx.npy'))

    source_mask = np.load(os.path.join(source, 'valid_mask.npy'))
    target_mask = np.load(os.path.join(target, 'valid_mask.npy'))

    corres = np.arange(source_xyz.shape[0])

    source_x_y = map_points_xy(source_idx, source_mask, source_xyz.shape[0])
    target_x_y = map_points_xy(target_idx, target_mask, target_xyz.shape[0])

    # mask = source_mask * target_mask
    mask = np.ones_like(corres).astype(bool)

    # corres = corres[mask]

    flow = build_flow(source_x_y, target_x_y, corres, mask)

    return flow


list_of_pairs = []
dt = np.dtype([('source', np.int32), ('target', np.int32), ('corres', 'object')])


def generate_synth_files(set_pcd):
    i = 0
    translation_range_x = 5
    for pcd_data in set_pcd:
        data = pcd_data["data"]
        convertor_fn(data, i)
        source_idx = i
        i += 1
        cloud_points, timestamp_data = np.hsplit(np.array(data), [3])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cloud_points)
        for j in range(30):
            translation_x = np.random.uniform(-translation_range_x, translation_range_x)
            translation_z = np.random.uniform(-1, 1)
            source_copy = copy.deepcopy(point_cloud)
            source_copy.translate([translation_x, 0, translation_z])
            new_data = np.hstack([np.asarray(source_copy.points), timestamp_data])
            convertor_fn(new_data, i)
            target_idx = i
            generate_synth_corres(source_idx, target_idx)
            # draw_registration_result(point_cloud, source_copy)
            i += 1


def generate_synth_corres(source_idx, target_idx):
    source_path = os.path.join(data_dir, str(source_idx))
    target_path = os.path.join(data_dir, str(target_idx))
    print(f"source: {source_path}, target: {target_path}")
    pair = dict()
    pair["source"] = source_idx
    pair["target"] = target_idx
    pair["corres_dir"] = f"corres_{source_idx}_{target_idx}"
    flow = synth_flow(source_path, target_path)
    print(np.count_nonzero(flow))
    flow_img = flow_to_color(flow.transpose(1, 2, 0))

    dir_path = os.path.join(corres_path, pair["corres_dir"])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.save(os.path.join(dir_path, "flow.npy"), flow)
    plt.imsave(os.path.join(dir_path, 'flow_img.png'), flow_img)

    list_of_pairs.append(pair)
    print(f"corres count at {source_idx}: {len(list_of_pairs)}")


def optical_flow_calculation_workflow():
    # threshold for distance between the poses
    pose_distance_lower = 0.01
    pose_distance_upper = 1

    # threshold for nearest neighbor for KD tree result
    nearest_neighbour_distance = 0.2

    # imu data
    ground_truth_imu = np.load("../dataset/test_new/ground_truth_pose.npy")

    for source_idx in range(len(ground_truth_imu)):
        dir_path = f"../dataset/test_new/default_leaf_csv_new{source_idx}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        pair_list = []

        # extracting position(x, y, z) from ground truth
        poses = ground_truth_imu[:, 1:4]

        # calculating L2 norm between poses
        pose_distances = np.linalg.norm((poses[source_idx] - poses), axis=1)

        # extract indices of the poses that is within the threshold
        # indices = np.where((pose_distances >= pose_distance_lower) & (pose_distances <= pose_distance_upper))
        # indices = np.where(pose_distances != 100000)
        indices = [[1]]
        for target_idx in indices[0]:
            csv_dict = dict()
            csv_dict["source_id"] = source_idx
            csv_dict["target_id"] = target_idx
            pose_distance = pose_distances[target_idx]
            # assert pose_distance <= pose_distance_upper
            csv_dict["distance"] = pose_distance

            source = f"../dataset/test_new/{source_idx}"
            target = f"../dataset/test_new/{target_idx}"

            source_valid_coord = np.load(os.path.join(source, 'initial_flow.npy'))

            source_range_img = np.load(os.path.join(source, 'range.npy'))
            target_range_img = np.load(os.path.join(target, 'range.npy'))

            source_frame = np.load(os.path.join(source, 'world_frame.npy'))
            target_frame = np.load(os.path.join(target, 'world_frame.npy'))

            source_point_idx = np.load(os.path.join(source, 'idx.npy'))
            target_point_idx = np.load(os.path.join(target, 'idx.npy'))

            source_mask = np.load(os.path.join(source, 'valid_mask.npy'))
            target_mask = np.load(os.path.join(target, 'valid_mask.npy'))

            # get index for all points in 2D domain
            source_x_y = map_points_xy(source_point_idx, source_mask, source_frame.shape[0])
            target_x_y = map_points_xy(target_point_idx, target_mask, target_frame.shape[0])

            # Build a KD-tree using target point cloud in world frame
            # corres, fitness = perform_icp(source_frame, target_frame)
            # s_idx = source_x_y[corres[:, 0]]
            #
            # t_idx = target_x_y[corres[:, 1]]
            tree = cKDTree(target_frame, leafsize=1)

            # Find the nearest neighbor of each point from source world frame in target world frame
            distances, corres = tree.query(source_frame, distance_upper_bound=np.inf)

            csv_dict["correspondence"] = len(corres)

            neighbor = distances != np.inf
            count = np.bincount(corres, weights=neighbor.astype(int))
            # count = np.bincount(corres)

            # find unique correspondences
            true_val = np.where((count == 1))[0]
            true_indices = [np.column_stack(np.where(corres == i)).ravel().tolist() for i in true_val]
            true_indices = np.array(true_indices)
            unique_mask = np.zeros_like(corres)
            unique_mask[true_indices] = 1

            # find non unique correspondences and assign the closest point among to the group to the respective
            # target point
            non_unique = np.where(count > 1)[0]
            non_unique_pair = [[i, np.column_stack(np.where(corres == i)).ravel().tolist()] for i in non_unique]
            for i, val in enumerate(non_unique_pair):
                min_distance_id = distances[val[1]].argmin()
                unique_mask[val[1][min_distance_id]] = 1

            # unique mask (index of all correspondence considered to be unique
            unique_mask = unique_mask.astype(bool)

            # correspondence that are within the threshold distance
            neighbour_mask = distances <= nearest_neighbour_distance

            csv_dict["neighbours"] = np.count_nonzero(neighbour_mask.astype(int))
            csv_dict["unique"] = np.count_nonzero(unique_mask.astype(int))

            valid_mask = neighbour_mask * unique_mask

            csv_dict["valid (neighbor and unique)"] = np.count_nonzero(valid_mask.astype(int))

            s_idx = source_x_y

            t_idx = target_x_y[corres]

            # calculate the flow (difference between target pixel index and source pixel index)
            diff_x = t_idx[:, 0] - s_idx[:, 0]
            diff_y = t_idx[:, 1] - s_idx[:, 1]

            # mask that masks flow result, that are nan in diff_x, nan in diff_y and not nearest neighbor and not unique
            mask = np.invert(np.isnan(diff_x)) & np.invert(np.isnan(diff_y)) & valid_mask

            csv_dict["valid diff_x"] = np.count_nonzero(np.invert(np.isnan(diff_x)).astype(int))
            csv_dict["valid diff_y"] = np.count_nonzero(np.invert(np.isnan(diff_y)).astype(int))
            csv_dict["flow_mask (all valid out of 64000 pixels)"] = np.count_nonzero(mask.astype(int))

            s_idx = s_idx[mask].astype(int)
            diff_x = diff_x[mask].astype(int)
            diff_y = diff_y[mask].astype(int)

            x = s_idx[:, 0]
            y = s_idx[:, 1]

            # create flow
            flow = np.zeros((2, 32, 2000))
            flow[0, x, y] = diff_x
            flow[1, x, y] = diff_y

            csv_dict["non zero flow along x (out of 64000 pixels)"] = np.count_nonzero(flow[0])
            csv_dict["non zero flow along x in percent"] = round(((np.count_nonzero(flow[0]) / 64000) * 100), 4)
            csv_dict["non zero flow along y (out of 64000 pixels)"] = np.count_nonzero(flow[1])
            csv_dict["non zero flow y in percent"] = round(((np.count_nonzero(flow[1]) / 64000) * 100), 4)

            pair_list.append(csv_dict)

            # plot_optical_flow(source_range_img, flow, mask)

            # plot_correspondence2(source_range_img, target_range_img, flow, source_valid_coord)

            # flow in image representation
            flow_img = flow_to_color(flow.transpose(1, 2, 0))
            plt.imsave(f"{dir_path}/flow_{source_idx}_{target_idx}.png", flow_img)

            source_w_frame = o3d.geometry.PointCloud()
            source_w_frame.points = o3d.utility.Vector3dVector(source_frame)

            target_w_frame = o3d.geometry.PointCloud()
            target_w_frame.points = o3d.utility.Vector3dVector(target_frame)

            o3d.io.write_point_cloud("../dataset/test_new/0/w_pcd_file.pcd", source_w_frame)
            o3d.io.write_point_cloud("../dataset/test_new/1/w_pcd_file.pcd", target_w_frame)

            print(f"processed source:{source_idx}, target:{target_idx}")

        # save the result for analytics purpose
        file_name = f"{dir_path}/csv_{source_idx}.csv"

        df = pd.DataFrame(pair_list)

        df.to_csv(file_name, index=False)


def plot_optical_flow(image1, flow, mask):
    h, w = image1.shape
    x, y = np.where((flow[0] != 0) | (flow[1] != 0))
    # _, y = np.where(flow[1] != 0)
    dx = flow[0, x, y]
    dy = flow[1, x, y]

    plt.figure(figsize=(10, 8))
    plt.imshow(image1, cmap='gray')

    # Scale the arrows for better visualization
    scale_factor = 5
    plt.quiver(x, y, dx * scale_factor, dy * scale_factor, color='red', angles='xy', scale_units='xy', scale=1)

    plt.title('Optical Flow Correspondence')
    plt.axis('off')
    plt.show()


def plot_correspondence2(image1, image2, flow, source_valid_coord):
    x, y = np.where((source_valid_coord[0] != -1) & (source_valid_coord[1] != -1))
    x, y = np.where((flow[0] != 0) | (flow[1] != 0))
    dx = flow[0, x, y]
    dy = flow[1, x, y]

    target_coords = np.copy(source_valid_coord)
    target_coords[0, x, y] += dx
    target_coords[1, x, y] += dy

    s_x, s_y = source_valid_coord[:, x, y]
    t_x, t_y = target_coords[:, x, y]

    # dest_x = (x + dx).astype(int)
    # dest_y = (y + dy).astype(int)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    axes[0].imshow(image1)
    axes[1].imshow(image2)

    # Draw a line connecting two specific points (e.g., first points of both plots)
    # line_s = (s_y.astype(int), s_x.astype(int))
    # line_t = (t_y.astype(int), t_x.astype(int))
    # line_color = 'r'
    # fig.lines.extend([plt.Line2D(line_x, line_y, color=line_color, linestyle='--')])
    for i in range(len(s_y)):
        print(i)
        line_s = (int(s_y[i]), int(s_x[i]))
        line_t = (int(t_y[i]), int(t_x[i]))
        con = ConnectionPatch(xyA=line_s, xyB=line_t, coordsA="data", coordsB="data",
                              axesA=axes[0], axesB=axes[1], color="red")
        axes[1].add_artist(con)

        axes[0].plot(line_s[0], line_s[1], 'ro', markersize=5)
        axes[1].plot(line_t[0], line_t[1], 'ro', markersize=5)

    plt.tight_layout()
    plt.show()


def plot_correspondence(image1, image2, flow):
    canvas = np.copy(image2)
    # Draw correspondence lines on the canvas
    x, y = np.where((flow[0] != 0) | (flow[1] != 0))
    # dx, dy = flow[x, y]
    dx = flow[0, x, y]
    dy = flow[1, x, y]
    end_point = ((x + dx).astype(int), (y + dy).astype(int))
    cv2.line(canvas, (x, y), end_point, (0, 255, 0), 1)

    # Plot the original image with the correspondence lines
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title('Correspondence Lines')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# lidar_to_imu = {'translation': [-0.001, -0.00855, 0.055], 'rotation': [0.7071068, -0.7071068, 0, 0]}
lidar_to_imu = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
lidar_param = {'height': 32, 'width': 2000, 'h_res': 0.18, 'v_res': 1, 'v_fov': {'down': -16, 'up': 15}}
data_dir = "../dataset/test_new"
corres_path = os.path.join(data_dir, "correspondence")
gt_gen_path = os.path.join(data_dir, "ground_truth_pose.npy")

lidar_data_converter = LidarDataConverter(
    lidar_param=lidar_param, lidar_2_imu=lidar_to_imu,
    save_dir=data_dir, generate_gt=True)


def convertor_fn(data, i):
    save_dir = lidar_data_converter(data, i)
    print(f"processed..{i}")
    return save_dir


def test_pipeline():
    i = 0
    list_gt = []
    bag = rosbag.Bag("../../data/hilti/rosbag/exp05_imu/exp05_construction_upper_level_2.bag")
    gt_file_path = "../dataset/test_new/fake_imu.txt"

    with open(gt_file_path) as file:
        ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])
    # ground_truth_imu = ground_truth_imu[ground_truth_imu[:, 0].argsort()]

    try:
        for topic, msg, time in bag.read_messages(topics=['/hesai/pandar']):
            if i == 1:
                # assert i < 1, "i should be less than length of gt file"
                # if ground_truth_imu[i][0] == time.to_time():
                data = list(pc2.read_points(msg, skip_nans=True,
                                            field_names=['x', 'y', 'z', 'timestamp']))

                source_dir = lidar_data_converter(data, 0, ground_truth_imu[0], ground_truth_imu[1])

                list_gt.append(ground_truth_imu[0])

                cloud_points, timestamp_data = np.hsplit(np.array(data), [3])
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(cloud_points)

                # rotate_radian = np.radians(rotate_degree)

                # rotation_matrix_y = np.array([[np.cos(rotate_radian), 0, np.sin(rotate_radian)],
                #                               [0, 1, 0],
                #                               [-np.sin(rotate_radian), 0, np.cos(rotate_radian)]
                #                               ])
                #
                # rotation_matrix_x = np.array([[1, 0, 0],
                #                               [0, np.cos(rotate_radian), -np.sin(rotate_radian)],
                #                               [0, np.sin(rotate_radian), np.cos(rotate_radian)]
                #                               ])

                target = copy.deepcopy(source)
                rotate_z = source.get_rotation_matrix_from_xyz((0, 0, np.pi / 6))
                print(rotate_z)
                target.rotate(rotate_z, center=(0, 0, 0))

                # target2 = copy.deepcopy(source)

                # target2.rotate(rotation_matrix_x)

                # draw_registration_result(source, target, target2)

                # source_dir = convertor_fn(data, 0)

                # target_data = copy.deepcopy(data)
                target_data = np.hstack([np.asarray(target.points), timestamp_data])
                target_dir = lidar_data_converter(target_data, 1, ground_truth_imu[2], ground_truth_imu[3])
                list_gt.append(ground_truth_imu[2])
                # target_dir = convertor_fn(target_data, 1)

                o3d.io.write_point_cloud("../dataset/test_new/0/pcd_file.pcd", source)
                o3d.io.write_point_cloud("../dataset/test_new/1/pcd_file.pcd", target)

                s_w = np.load(os.path.join(source_dir, "world_frame.npy"))
                t_w = np.load(os.path.join(target_dir, "world_frame.npy"))
                source_w = o3d.geometry.PointCloud()
                source_w.points = o3d.utility.Vector3dVector(s_w)
                target_w = o3d.geometry.PointCloud()
                target_w.points = o3d.utility.Vector3dVector(t_w)

                o3d.io.write_point_cloud("../dataset/test_new/0/world_frame.pcd", source_w)
                o3d.io.write_point_cloud("../dataset/test_new/1/world_frame.pcd", target_w)

                np.save(gt_gen_path, np.array(list_gt))
                print(source_dir, target_dir)

                # draw_registration_result(source, target)
            i = i + 1

    except Exception as e:
        print(f"Exception: {e}")


from PIL import Image


def generate_synth_corres2():
    source_path = "../dataset/test/0"
    target_path = "../dataset/test/1"
    range1 = np.load(os.path.join(source_path, "range.npy"))
    range2 = np.load(os.path.join(target_path, "range.npy"))
    pair = dict()
    pair["source"] = 0
    pair["target"] = 1
    pair["corres_dir"] = f"corres_{0}_{1}"
    flow, flow_img = utils.data_conversion.synth_flow(source_path, target_path)
    dir_path = "../dataset/test/correspondence"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.save(os.path.join(dir_path, "flow.npy"), flow)
    image = Image.fromarray(flow_img)
    image.save(os.path.join(dir_path, 'flow_img.png'))
    plt.imsave(os.path.join(dir_path, "range1.png"), range1)
    plt.imsave(os.path.join(dir_path, "range2.png"), range2)
    list_of_pairs.append(pair)
    print(f"corres count at {0}: {len(list_of_pairs)}")


def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


def octree_nearest_neighbor():
    source = f"../dataset/test2/{0}"
    target = f"../dataset/test2/{1}"
    source_frame = np.load(os.path.join(source, 'world_frame.npy'))
    target_frame = np.load(os.path.join(target, 'world_frame.npy'))
    source_w_frame = o3d.geometry.PointCloud()
    source_w_frame.points = o3d.utility.Vector3dVector(source_frame)
    target_w_frame = o3d.geometry.PointCloud()
    target_w_frame.points = o3d.utility.Vector3dVector(target_frame)

    target_octree = o3d.geometry.Octree(max_depth=4)
    target_octree.convert_from_point_cloud(target_w_frame, size_expand=0.01)
    pp = target_octree.locate_leaf_node(source_frame[0])
    print(pp)
    # target_octree.traverse(f_traverse)


if __name__ == "__main__":
    # Call the main function when the script is run
    # get_flow_matrix()
    # view_best_distance_flow(2)
    # pcd_set = read_bag()
    # os.makedirs(corres_path)
    # generate_synth_files(pcd_set)
    # arr = np.empty(len(list_of_pairs), dtype=dt)
    # for idx, val in enumerate(list_of_pairs):
    #    arr[idx] = (val['source'], val['target'], val['corres_dir'])
    # np.save(os.path.join(data_dir, "corres.npy"), arr)
    # target_idx = view_best_distance_idx(0)

    # source_frame = np.load(os.path.join(source_path, 'xyz.npy'))
    # target_frame = np.load(os.path.join(target_path, 'xyz.npy'))
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(source_frame)

    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(target_frame)
    # draw_registration_result(pcd1, pcd2)
    # flow = get_pixel_match(source_path, target_path, 0.2)
    # optical_flow_calculation_workflow()
    # test_pipeline()
    # generate_synth_corres2()
    optical_flow_calculation_workflow()
    # octree_nearest_neighbor()
    print("flow")
