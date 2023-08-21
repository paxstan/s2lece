import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import copy
from utils.data_conversion import LidarDataConverter, get_pixel_match, synth_flow
import logging
import open3d as o3d
from PIL import Image
import pandas as pd


class DatasetCreator:
    def __init__(self, config):
        self.config = config
        self.dataset = config["dataset"][config["datasets"][config["dataset_choice"]]]
        self.root = self.dataset['data_dir']
        self.generate_ground_truth = self.dataset["generate_ground_truth"]
        self.ros_bag = os.path.join(self.dataset['path'], self.dataset['ros_bag'])
        self.ground_truth = os.path.join(self.dataset['path'], self.dataset['ground_truth'])

        if self.generate_ground_truth:
            with open(self.ground_truth) as file:
                self.ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])
            self.ground_truth_imu = self.ground_truth_imu[self.ground_truth_imu[:, 0].argsort()]

        self.lidar_data_converter = LidarDataConverter(
            lidar_param=config["lidar_param"], lidar_type="hilti",
            save_dir=self.dataset["data_dir"], generate_gt=self.generate_ground_truth)
        self.gt_gen_path = os.path.join(self.dataset['data_dir'], "ground_truth_pose.npy")

        self.correspondence_param = self.config["correspondence"]
        self.corres_path = os.path.join(self.dataset['data_dir'], "correspondence")

    def __call__(self):
        if not os.path.exists(self.root):
            logging.info("extracting raw dataset from ros bag.....")
            os.makedirs(self.root)
            os.makedirs(self.corres_path)
            self.point_cloud_extractor()
            if self.generate_ground_truth:
                logging.info("Generating pairs....")
                self.generate_pairs()
            logging.info("Finished extraction of raw dataset !")
        else:
            logging.info("raw dataset already extracted !")
        return os.path.abspath(self.root)

    def point_cloud_extractor(self):
        i = 0
        list_gt = []
        bag = rosbag.Bag(self.ros_bag)
        try:
            for topic, msg, time in bag.read_messages(topics=['/hesai/pandar']):
                data = list(pc2.read_points(msg, skip_nans=True, field_names=['x', 'y', 'z', 'timestamp']))

                if self.generate_ground_truth:

                    assert i < 20, "i should be less than length of gt file"  # len(self.ground_truth_imu)-1
                    if self.ground_truth_imu[i][0] == time.to_time():
                        # process and save point cloud in the form of npy file
                        self.lidar_data_converter(data, i, self.ground_truth_imu[i], self.ground_truth_imu[i + 1])

                        list_gt.append(self.ground_truth_imu[i])

                        print(f"processed... {i}")
                        i = i + 1
                else:
                    assert i < 20, "i should be less than length of gt file"
                    # process and save point cloud in the form of npy file
                    self.lidar_data_converter(data, i)

                    print(f"processed... {i}")
                    i = i + 1

        except AssertionError as error:
            logging.error(f"i : {i}, error : {error}")

        np.save(self.gt_gen_path, np.array(list_gt))
        bag.close()
        logging.info("Point cloud extraction from ROS Bag completed....")

    def generate_pairs(self):
        list_of_pairs = []
        dt = np.dtype([('source', np.int32), ('target', np.int32), ('corres', 'object')])
        poses = np.load(self.gt_gen_path)
        pair_threshold = self.correspondence_param["pair_distance"]
        for i, pose in enumerate(poses):
            try:
                # pair_dict = dict()
                # pair_dict["source"] = i
                source_path = os.path.join(self.root, str(i))

                distances = np.linalg.norm((pose - poses), axis=1)

                index = np.where((distances >= pair_threshold[0]) & (distances <= pair_threshold[1]))
                for ix in index[0]:
                    try:
                        target_path = os.path.join(self.root, str(ix))
                        pair = dict()
                        pair["source"] = i
                        pair["target"] = ix
                        pair["corres_dir"] = f"corres_{i}_{ix}"

                        flow = get_pixel_match(source_path, target_path, self.correspondence_param["nearest_neighbor"])

                        dir_path = os.path.join(self.corres_path, pair["corres_dir"])
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        np.save(os.path.join(dir_path, "flow.npy"), flow)

                        list_of_pairs.append(pair)

                    except FileNotFoundError as e:
                        print(e)
                logging.info(f"corres count at {i}: {len(list_of_pairs)}")
            except FileNotFoundError as e:
                logging.error(e)

        arr = np.empty(len(list_of_pairs), dtype=dt)
        for idx, val in enumerate(list_of_pairs):
            arr[idx] = (val['source'], val['target'], val['corres_dir'])
        np.save(os.path.join(self.root, "corres.npy"), arr)
        logging.info("Finished generating pairs....")


class SyntheticDatasetCreator:
    def __init__(self, config):
        self.config = config
        self.dataset = config["dataset"][config["datasets"][config["dataset_choice"]]]
        self.root = self.dataset['data_dir']
        self.ros_bag = os.path.join(self.dataset['path'], self.dataset['ros_bag'])
        self.generate_ground_truth = self.dataset["generate_ground_truth"]
        self.ground_truth = os.path.join(self.dataset['path'], self.dataset['ground_truth'])
        self.lidar_param = config["lidar_param"]["kitti"]

        if self.generate_ground_truth:
            with open(self.ground_truth) as file:
                self.ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])
            self.ground_truth_imu = self.ground_truth_imu[self.ground_truth_imu[:, 0].argsort()]

        self.lidar_data_converter = LidarDataConverter(
            lidar_param=self.lidar_param, lidar_type="hilti",
            save_dir=self.dataset["data_dir"], generate_gt=self.generate_ground_truth)

        self.correspondence_param = self.config["correspondence"]
        self.corres_path = os.path.join(self.dataset['data_dir'], "correspondence")
        self.translation_range_x = 5
        self.translation_range_y = 5
        self.translation_range_z = 5
        self.list_of_pairs = []
        self.dt = np.dtype([('source', np.int32), ('target', np.int32), ('corres', 'object')])

    def __call__(self, *args, **kwargs):
        if not os.path.exists(self.root):
            logging.info("extracting raw dataset from ros bag.....")
            os.makedirs(self.root)
            os.makedirs(self.corres_path)
            self.point_cloud_extractor()
            arr = np.empty(len(self.list_of_pairs), dtype=self.dt)
            for idx, val in enumerate(self.list_of_pairs):
                arr[idx] = (val['source'], val['target'], val['corres_dir'])
            np.save(os.path.join(self.root, "corres.npy"), arr)
            logging.info("Finished extraction of raw dataset !")
        else:
            logging.info("raw dataset already extracted !")
        return os.path.abspath(self.root)

    def convertor_fn(self, data, i, pose1, pose2):
        # save_dir = self.lidar_data_converter(data, i)
        save_dir = self.lidar_data_converter(data, i, pose1, pose2)
        return save_dir

    def point_cloud_extractor(self):
        i = 0
        source_idx = 0
        target_idx = 1
        bag = rosbag.Bag(self.ros_bag)
        try:
            for topic, msg, time in bag.read_messages(topics=['/hesai/pandar']):
                if self.generate_ground_truth:
                    print("msg count: ", i)
                    assert i < len(self.ground_truth_imu) - 1, "i should be less than length of gt file"
                    if self.ground_truth_imu[i][0] == time.to_time():
                        data = list(pc2.read_points(msg, skip_nans=True, field_names=['x', 'y', 'z', 'timestamp']))
                        # process and save point cloud in the form of npy file
                        self.convertor_fn(data, source_idx, self.ground_truth_imu[i], self.ground_truth_imu[i + 1])
                        cloud_points, timestamp_data = np.hsplit(np.array(data), [3])
                        point_cloud = o3d.geometry.PointCloud()
                        point_cloud.points = o3d.utility.Vector3dVector(cloud_points)
                        for j in range(2):
                            target_idx += j
                            source_copy = copy.deepcopy(point_cloud)
                            # translation_x = np.random.uniform(-self.translation_range_x, self.translation_range_x)
                            # translation_y = np.random.uniform(-self.translation_range_y, self.translation_range_y)
                            # translation_z = np.random.uniform(-self.translation_range_z, self.translation_range_z)
                            # source_copy.translate([translation_x, translation_y, translation_z])
                            rotate_z = np.random.uniform(-30, 30)
                            rotation = source_copy.get_rotation_matrix_from_xyz((0, 0, rotate_z))
                            source_copy.rotate(rotation, center=(0, 0, 0))
                            new_data = np.hstack([np.asarray(source_copy.points), timestamp_data])
                            self.convertor_fn(new_data, target_idx)
                            self.generate_synth_corres(source_idx, target_idx, str(rotate_z))

                        source_idx = target_idx + 1
                        target_idx = source_idx + 1
                i = i + 1

        except AssertionError as error:
            logging.error(f"i : {source_idx}, error : {error}")

        bag.close()
        logging.info("Point cloud extraction from ROS Bag completed....")

    def generate_synth_corres(self, source_idx, target_idx, transform_param):
        source_path = os.path.join(self.root, str(source_idx))
        target_path = os.path.join(self.root, str(target_idx))
        pair = dict()
        pair["source"] = source_idx
        pair["target"] = target_idx
        pair["transform_param"] = transform_param
        pair["corres_dir"] = f"corres_{source_idx}_{target_idx}"
        flow, flow_img = synth_flow(source_path, target_path)
        dir_path = os.path.join(self.corres_path, pair["corres_dir"])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(os.path.join(dir_path, "flow.npy"), flow)
        image = Image.fromarray(flow_img)
        image.save(os.path.join(dir_path, 'flow_img.png'))

        f = open(os.path.join(dir_path, "tranform_param.txt"), "a")
        f.write(f"z rotation: {transform_param}")
        f.close()

        self.list_of_pairs.append(pair)
        print(f"corres count at {source_idx}: {len(self.list_of_pairs)}")


class KittiDatasetCreator:
    def __init__(self, config):
        self.config = config
        self.dataset = config["dataset"][config["datasets"][config["dataset_choice"]]]
        self.root = self.dataset['data_dir']
        self.dataset_path = os.path.join(self.dataset['path'], self.dataset['dataset_path'])
        self.generate_ground_truth = self.dataset["generate_ground_truth"]
        self.ground_truth = os.path.join(self.dataset['path'], self.dataset['ground_truth'])
        self.lidar_param = config["lidar_param"]["kitti"]

        if self.generate_ground_truth:
            with open(self.ground_truth) as file:
                self.ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])
            # self.ground_truth_imu = self.ground_truth_imu[1:]

        self.lidar_data_converter = LidarDataConverter(
            lidar_param=self.lidar_param, save_dir=self.dataset["data_dir"],
            generate_gt=self.generate_ground_truth, lidar_type="kitti")
        self.gt_gen_path = os.path.join(self.dataset['data_dir'], "ground_truth_pose.npy")

        self.correspondence_param = self.config["correspondence"]
        self.corres_path = os.path.join(self.dataset['data_dir'], "correspondence")

    def __call__(self):
        if not os.path.exists(self.root):
            logging.info("extracting raw dataset from ros bag.....")
            os.makedirs(self.root)
            os.makedirs(self.corres_path)
            self.point_cloud_extractor()
            self.generate_pairs()
            logging.info("Finished extraction of raw dataset !")
        else:
            logging.info("raw dataset already extracted !")
        return os.path.abspath(self.root)

    def point_cloud_extractor(self):
        list_gt = []
        for i, pose in enumerate(self.ground_truth_imu):
            pcd_path = os.path.join(self.dataset_path, f"{i:06d}.pcd")
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            timestamp = np.full((points.shape[0], 1), pose[0], dtype=float)
            data = np.hstack([points, timestamp])
            self.lidar_data_converter(data, i, self.ground_truth_imu[i], None)
            list_gt.append(self.ground_truth_imu[i])
            print(f"processed... {i}")
        np.save(self.gt_gen_path, np.array(list_gt))
        logging.info("Point cloud extraction completed....")

    def generate_pairs(self):
        list_of_pairs = []
        dt = np.dtype([('source', np.int32), ('target', np.int32), ('corres', 'object')])
        # poses = np.load(self.gt_gen_path)[:, 1:4]
        timestamps = np.load(self.gt_gen_path)[:, 0]
        pair_threshold = self.correspondence_param["pair_distance"]
        # for i in range(poses.shape[0] - 1):
        for i, time in enumerate(timestamps):
            try:
                # source_path = os.path.join(self.root, str(i))

                # distances = np.linalg.norm((pose - poses), axis=1)
                time_diff = abs(timestamps - time)

                # index = np.where((distances >= pair_threshold[0]) & (distances <= pair_threshold[1]))
                index = np.where((time_diff > 0.0) & (time_diff < 0.5))
                for ix in index[0]:
                    pair = dict()
                    pair["source"] = i
                    source_path = os.path.join(self.root, str(i))
                    target_path = os.path.join(self.root, str(ix))
                    pair["target"] = ix
                    pair["corres_dir"] = f"corres_{pair['source']}_{pair['target']}"

                    flow, flow_img, csv_dict = get_pixel_match(source_path, target_path,
                                                               self.correspondence_param["nearest_neighbor"],
                                                               height=self.lidar_param["height"],
                                                               width=self.lidar_param["width"])

                    dir_path = os.path.join(self.corres_path, pair["corres_dir"])
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    np.save(os.path.join(dir_path, "flow.npy"), flow)
                    image = Image.fromarray(flow_img)
                    image.save(os.path.join(dir_path, 'flow_img.png'))

                    file_name = os.path.join(dir_path, f"csv_{pair['corres_dir']}.csv")

                    df = pd.DataFrame(csv_dict)

                    df.to_csv(file_name, index=False)

                    list_of_pairs.append(pair)

                logging.info(f"corres count at {i}: {len(list_of_pairs)}")
            except Exception as e:
                logging.info(f"generate_pair: {e}")

        arr = np.empty(len(list_of_pairs), dtype=dt)
        for idx, val in enumerate(list_of_pairs):
            arr[idx] = (val['source'], val['target'], val['corres_dir'])
        np.save(os.path.join(self.root, "corres.npy"), arr)
        logging.info("Finished generating pairs....")
