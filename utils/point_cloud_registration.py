import open3d as o3d
import numpy as np
from utils.plot_tools import draw_point_cloud_pair

VOXEL_SIZE = 0.05  # means 5cm for the dataset
SAMPLE_TRANSFORM = np.array(
    [9.99999806e-01, -5.97747650e-04, 1.73649736e-04, 1.25537628e-02,
     5.97739542e-04, 9.99999820e-01, 4.67437736e-05, -9.54559376e-03,
     -1.73677646e-04, -4.66399672e-05, 9.99999984e-01, -1.74009819e-07,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]).reshape(4, 4)


class PointCloudRegistration:
    def __init__(self, voxel_size=VOXEL_SIZE):
        self.voxel_size = voxel_size

    def preprocess_point_cloud(self, pcd):
        print(":: Downsample with a voxel size %.3f." % self.voxel_size)
        pcd_down = pcd.voxel_down_sample(self.voxel_size)

        radius_normal = self.voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        distance_threshold = self.voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % self.voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=source_down, target=target_down,
            source_feature=source_fpfh, target_feature=target_fpfh,
            mutual_filter=True, max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4, checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999))
        return result

    def refine_registration(self, source, target, result_ransac):
        distance_threshold = self.voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_icp(
            source=source, target=target, max_correspondence_distance=distance_threshold,
            init=result_ransac, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        )
        return result

    def __call__(self, source, target):
        source_down, source_fpfh = self.preprocess_point_cloud(source)
        target_down, target_fpfh = self.preprocess_point_cloud(target)

        result_ransac = self.execute_global_registration(source_down, target_down,
                                                         source_fpfh, target_fpfh)
        # draw_point_cloud_pair(source_down, target_down, result_ransac.transformation)

        result_icp = self.refine_registration(source_down, target_down, result_ransac.transformation)
        # draw_point_cloud_pair(source, target, result_icp.transformation)
        return result_icp

#
# #
# if __name__ == "__main__":
#     icp_reg = PointCloudRegistration()
#     source = o3d.io.read_point_cloud(
#         "../old.pcd")
#     target = o3d.io.read_point_cloud(
#         "../transformed.pcd")
#     icp_reg(source, target)
