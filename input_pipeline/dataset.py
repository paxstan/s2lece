import copy
import os
import open3d as o3d
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from PIL import Image
from utils.data_conversion import pcd_transformation, LidarDataConverter, interpolated_data, perform_icp
from input_pipeline.dataset_classes import LidarData

TRANSLATION_LIDAR_IMU = np.array([-0.001, -0.00855, 0.055])
ROTATION_LIDAR_IMU = np.array([0.7071068, -0.7071068, 0, 0])
TRANSLATION_IMU_WORLD = np.array([-3.45, 6.27, 1.79])
ROTATION_IMU_WORLD = np.array([0.6602272584, 0.7510275859, 0.003393373643, -0.006783616141])


class DatasetCreator:
    def __init__(self, config):
        self.config = config
        with open(config['ground_truth_path']) as file:
            self.ground_truth_imu = [line.rstrip().split(" ") for line in file]
        self.lidar_data_converter = LidarDataConverter(save_dir=self.config['data_dir'])
        self.list_of_pcd = []
        self.lidar_data_list = []

    def __call__(self):
        if not os.path.exists(self.config['data_dir']):
            print("extracting raw dataset from ros bag.....")
            os.makedirs(self.config['data_dir'])
            self.point_cloud_extractor()
            print("Finished extraction of raw dataset !")
        else:
            print("raw dataset already extracted !")

        self.process_data()
        self.generate_correspondence()

    def point_cloud_extractor(self):
        i = 0
        bag = rosbag.Bag(self.config['ros_bag_path'])
        for topic, msg, time in bag.read_messages(topics=['/hesai/pandar']):
            if float(self.ground_truth_imu[i][0]) == time.to_time():
                data = list(pc2.read_points(msg, skip_nans=True,
                                            field_names=['x', 'y', 'z', 'intensity', 'timestamp']))

                # process and save point cloud in the form of npy file
                self.lidar_data_converter(data, i)

                i = i + 1
        bag.close()

    def process_data(self):
        i = 0
        dir_list = os.walk(self.config['data_dir'])
        for sub_dir in os.listdir(self.config['data_dir']):
            sub_dir_path = os.path.join(self.config['data_dir'], sub_dir)
            lidar_data = LidarData(sub_dir_path)
            # org_data = np.load(f'{sub_dir_path}/org_data.npy')
            # range_data = np.load(f'{sub_dir_path}/range.npy')
            # xyz_data = np.load(f'{sub_dir_path}/xyz.npy')

            # extract point cloud
            cloud_data = np.asarray([data[:3] for data in lidar_data.org_data])
            l_pcd = o3d.geometry.PointCloud()
            l_pcd.points = o3d.utility.Vector3dVector(cloud_data)
            # transform from lidar frame to imu frame
            i_pcd = pcd_transformation(copy.deepcopy(l_pcd), ROTATION_LIDAR_IMU, TRANSLATION_LIDAR_IMU)

            # transform from imu frame to world frame
            translation_imu_world = [float(axis) for axis in self.ground_truth_imu[i][1:4]]
            rotation_imu_world = [float(quat) for quat in self.ground_truth_imu[i][4:]]
            world_pcd = pcd_transformation(copy.deepcopy(i_pcd), rotation_imu_world, translation_imu_world)

            self.list_of_pcd.append(i_pcd)
            self.lidar_data_list.append(lidar_data)

            # extract timedata
            # timestamp_data = [data[4] for data in msg_data]
            # list_points = [i for i, p_time in enumerate(timestamp_data) if p_time < float(ground_truth[i][0])]
            #
            # print("ground truth time- ", float(ground_truth[i][0]))
            # if float(ground_truth[i][0]) >= time.to_time():
            #     print("msg time- ", time.to_time())
            #     list_of_pcd.append(l_pcd)
            # else:
            #     print("outbound msg time- ", time.to_time())
            #     interp_rotation, interp_translation = interpolated_data(list_of_pcd)
            #     interp_pcd = pcd_transformation(list_of_pcd[0], interp_rotation, interp_translation)
            #     list_interp_pcd.append(interp_pcd)
            #     points = np.array(cloud_data)
            #     list_of_pcd = [l_pcd]
    def generate_correspondence(self):
        i = 0
        while i < len(self.list_of_pcd)-1:
            icp_result_list = perform_icp([self.list_of_pcd[i], self.list_of_pcd[i+1]])
            print(icp_result_list[0].fitness)
            for corres in np.asarray(icp_result_list[0].correspondence_set):
                pcd_a_index = np.where(self.lidar_data_list[i].xyz_data == corres[0])
                pcd_b_index = np.where(self.lidar_data_list[i+1].xyz_data == corres[1])
                print("correspondence pixel, a: ", pcd_a_index, " b: ", pcd_b_index)




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
