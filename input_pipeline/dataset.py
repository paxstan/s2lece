import copy
import os
import open3d as o3d
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from PIL import Image
from utils.data_conversion import pcd_transformation, LidarDataConverter, Imu2World, perform_kdtree
from input_pipeline.dataset_classes import LidarData

TRANSLATION_LIDAR_IMU = np.array([-0.001, -0.00855, 0.055])
ROTATION_LIDAR_IMU = np.array([0.7071068, -0.7071068, 0, 0])


class DatasetCreator:
    def __init__(self, config):
        self.config = config
        with open(config['ground_truth_path']) as file:
            self.ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])
        self.ground_truth_imu = self.ground_truth_imu[self.ground_truth_imu[:, 0].argsort()]
        self.new_gt = np.copy(self.ground_truth_imu)
        self.lidar_data_converter = LidarDataConverter(save_dir=self.config['data_dir'])
        # self.list_pair_dict = []

    def __call__(self):
        if not os.path.exists(self.config['data_dir']):
            print("extracting raw dataset from ros bag.....")
            os.makedirs(self.config['data_dir'])
            self.point_cloud_extractor()
            self.generate_pairs(threshold=0.2)
            print("Finished extraction of raw dataset !")
        else:
            print("raw dataset already extracted !")
        return self.load_data()

    def point_cloud_extractor(self):
        i = 0
        list_gt = []
        bag = rosbag.Bag(self.config['ros_bag_path'])
        try:
            for topic, msg, time in bag.read_messages(topics=['/hesai/pandar']):
                assert i < len(self.ground_truth_imu)-1, "i should be less than length of gt file"
                if self.ground_truth_imu[i][0] == time.to_time():
                    data = list(pc2.read_points(msg, skip_nans=True,
                                                field_names=['x', 'y', 'z', 'intensity', 'timestamp']))

                    # process and save point cloud in the form of npy file
                    self.lidar_data_converter(data, i, [self.ground_truth_imu[i], self.ground_truth_imu[i + 1]])
                    list_gt.append(self.ground_truth_imu[i])

                    print(f"processed... {i}")
                    i = i + 1
        except AssertionError as error:
            print(error)

        np.save(f"{self.config['data_dir']}/ground_truth_pose.npy", np.array(list_gt))
        bag.close()

    def generate_pairs(self, threshold):
        list_of_pairs = []
        dt = np.dtype([('source', np.int32), ('target', np.int32), ('corres', 'object')])
        poses = self.ground_truth_imu[:, 1:4]
        for i, pose in enumerate(poses):
            try:
                pair_dict = dict()
                pair_dict["source"] = i
                source_scan_path = os.path.join(self.config['data_dir'], str(i))
                source_scan_world_frame = np.load(f'{source_scan_path}/world_frame.npy')
                distances = np.linalg.norm((pose - poses), axis=1)
                index = np.where((distances > 0) & (distances <= 0.5))
                for ix in index[0]:
                    try:
                        target_scan_path = os.path.join(self.config['data_dir'], str(ix))
                        target_scan_world_frame = np.load(f'{target_scan_path}/world_frame.npy')
                        distances, indices = perform_kdtree(
                            source_scan_world_frame, target_scan_world_frame, threshold=threshold)
                        indices[distances == np.inf] = -1
                        if len(indices[distances != np.inf]) >= len(source_scan_world_frame) * 0.5:
                            pair = dict()
                            pair["source"] = i
                            pair["target"] = ix
                            pair["corres"] = indices
                            list_of_pairs.append(pair)
                    except FileNotFoundError as e:
                        print(e)
                print(f"corres count at {i}: {len(list_of_pairs)}")
            except FileNotFoundError as e:
                print(e)
        arr = np.empty(len(list_of_pairs), dtype=dt)
        for idx, val in enumerate(list_of_pairs):
            arr[idx] = (val['source'], val['target'], val['corres'])
        np.save(os.path.join(self.config['data_dir'], "corres.npy"), arr)

    def load_data(self):
        i = 0
        list_data = []
        folder_count = len(os.listdir(self.config['data_dir']))
        while i < folder_count:
            path = os.path.join(self.config['data_dir'], str(i))
            list_data.append(LidarData(path))
            i = i + 1
        return list_data

    def join_corres(self):
        i = 0
        folder_count = len(os.listdir(self.config['data_dir']))
        dt = np.dtype([('source', np.int32), ('target', np.int32), ('corres', 'object')])
        while i < folder_count:
            path = os.path.join(self.config['data_dir'], str(i))
            corres = np.load(f'{path}/corres.npy')


def random_image_loader(data_dir):
    # random_id = random.randint(1, 2277)
    random_id = 1
    range_array = np.load(f'{data_dir}/{random_id}/range.npy')
    # intensity_array = np.load(f'{data_dir}/{random_id}/intensity.npy')
    reflectivity_array = np.zeros(range_array.shape)
    mask_array = np.load(f'{data_dir}/{random_id}/valid_mask.npy')

    range_im = ((range_array / 8. + 1.) / 2. * 255).astype(np.uint8)
    # intensity_im = ((intensity_array / 8. + 1.) / 2. * 255).astype(np.uint8)
    reflectivity_im = ((reflectivity_array / 8. + 1.) / 2. * 255).astype(np.uint8)

    # img_stack = np.stack((range_im, intensity_im, reflectivity_im), axis=2)

    # stacked_img = Image.fromarray(img_stack, mode="RGB")
    img = Image.fromarray(range_im)

    return img, mask_array
