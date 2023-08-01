import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from PIL import Image
from utils.data_conversion import LidarDataConverter, get_pixel_match
import logging


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
            lidar_param=config["lidar_param"], lidar_2_imu=config["lidar_to_imu"],
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

                    assert i < len(self.ground_truth_imu)-1, "i should be less than length of gt file"  # len(self.ground_truth_imu)-1
                    if self.ground_truth_imu[i][0] == time.to_time():

                        # process and save point cloud in the form of npy file
                        self.lidar_data_converter(data, i, self.ground_truth_imu[i], self.ground_truth_imu[i + 1])

                        list_gt.append(self.ground_truth_imu[i])

                        print(f"processed... {i}")
                        i = i + 1
                else:
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
        poses = self.ground_truth_imu[:, 1:4]
        pair_threshold = self.correspondence_param["pair_distance"]
        for i, pose in enumerate(poses):
            try:
                pair_dict = dict()
                pair_dict["source"] = i
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
