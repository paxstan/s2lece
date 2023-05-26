import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from PIL import Image
from utils.data_conversion import LidarDataConverter, perform_kdtree
from input_pipeline.dataset import get_pixel_match


class DatasetCreator:
    def __init__(self, config):
        self.config = config
        self.root = self.config['data_dir']
        self.corres_path = os.path.join(self.config['data_dir'], "correspondence")
        self.gt_path = os.path.join(self.config['data_dir'], "ground_truth_pose.npy")
        with open(config['ground_truth_path']) as file:
            self.ground_truth_imu = np.array([tuple(map(float, line.rstrip().split(" "))) for line in file])
        self.ground_truth_imu = self.ground_truth_imu[self.ground_truth_imu[:, 0].argsort()]
        self.new_gt = np.copy(self.ground_truth_imu)
        self.lidar_data_converter = LidarDataConverter(
            lidar_param=config["lidar_param"], lidar_2_imu=config["lidar_to_imu"], save_dir=config["data_dir"])

    def __call__(self):
        if not os.path.exists(self.root):
            print("extracting raw dataset from ros bag.....")
            os.makedirs(self.root)
            os.makedirs(self.corres_path)
            self.point_cloud_extractor()
            self.generate_pairs(threshold=0.2)
            print("Finished extraction of raw dataset !")
        else:
            print("raw dataset already extracted !")

    def point_cloud_extractor(self):
        i = 0
        list_gt = []
        bag = rosbag.Bag(self.config['ros_bag_path'])
        try:
            for topic, msg, time in bag.read_messages(topics=['/hesai/pandar']):
                assert i < 21, "i should be less than length of gt file"
                if self.ground_truth_imu[i][0] == time.to_time():
                    data = list(pc2.read_points(msg, skip_nans=True,
                                                field_names=['x', 'y', 'z', 'timestamp']))

                    # process and save point cloud in the form of npy file
                    self.lidar_data_converter(data, i, self.ground_truth_imu[i], self.ground_truth_imu[i + 1])
                    list_gt.append(self.ground_truth_imu[i])

                    print(f"processed... {i}")
                    i = i + 1
        except AssertionError as error:
            print(error)

        np.save(self.gt_path, np.array(list_gt))
        bag.close()

    def generate_pairs(self, threshold):
        list_of_pairs = []
        dt = np.dtype([('source', np.int32), ('target', np.int32), ('corres', 'object')])
        poses = self.ground_truth_imu[:, 1:4]
        for i, pose in enumerate(poses):
            try:
                pair_dict = dict()
                pair_dict["source"] = i
                source_path = os.path.join(self.root, str(i))
                source_world_frame = np.load(f'{source_path}/world_frame.npy')
                source_img = np.load(f'{source_path}/range.npy')

                distances = np.linalg.norm((pose - poses), axis=1)
                index = np.where((distances > 0) & (distances <= 0.5))
                for ix in index[0]:
                    try:
                        target_path = os.path.join(self.root, str(ix))
                        target_world_frame = np.load(f'{target_path}/world_frame.npy')
                        target_mask = np.load(f'{target_path}/valid_mask.npy')

                        distances, indices = perform_kdtree(
                            source_world_frame, target_world_frame, threshold=threshold)
                        indices[distances == np.inf] = -1
                        if len(indices[distances != np.inf]) >= len(target_world_frame) * 0.5:
                            pair = dict()
                            pair["source"] = i
                            pair["target"] = ix
                            pair["corres_dir"] = f"corres_{i}_{ix}"

                            # get flow according to pair index and reprojected mask2
                            flow, mask_valid_in_2 = get_pixel_match(
                                target_mask, source_path, target_path, indices, source_img.shape)

                            dir_path = os.path.join(self.corres_path, pair["corres_dir"])
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)
                            np.save(os.path.join(dir_path, "flow.npy"), flow)
                            np.save(os.path.join(dir_path, "mask_valid_2.npy"), mask_valid_in_2)

                            list_of_pairs.append(pair)
                    except FileNotFoundError as e:
                        print(e)
                print(f"corres count at {i}: {len(list_of_pairs)}")
            except FileNotFoundError as e:
                print(e)

        arr = np.empty(len(list_of_pairs), dtype=dt)
        for idx, val in enumerate(list_of_pairs):
            arr[idx] = (val['source'], val['target'], val['corres_dir'])
        np.save(os.path.join(self.root, "corres.npy"), arr)


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
