import numpy as np
from absl import app, flags
from input_pipeline.dataset_creator import random_image_loader, DatasetCreator
from input_pipeline.dataset import RealPairDataset, SyntheticPairDataset
from input_pipeline.dataloader import PairLoader, threaded_loader, SythPairLoader
from input_pipeline.preprocessing import RandomScale, RandomTilting, PixelNoise, RandomTranslation, RandomCrop
import yaml
from visualization.visualization import show_flow
from utils import common
from utils.train import MyTrainer, TestTrainer
from models.patchnet import Quad_L2Net_ConfCFS
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.sampler import NghSampler2
from utils import utils_params
from models.losses import *
from models.reliability_loss import ReliabilityLoss
from models.repeatability_loss import CosimLoss, PeakyLoss
import torch
import torch.optim as optim
import extract
import matching

RANDOM_SCALE = RandomScale(min_size=80, max_size=128, can_upscale=True)
RANDOM_TILT = RandomTilting(magnitude=0.025, directions="left")
RANDOM_NOISE = PixelNoise(ampl=50)
RANDOM_TRANS = RandomTranslation(roll=100)
RANDOM_RESCALE = RandomScale(64, 64, can_upscale=True)
RANDOM_CROP = RandomCrop((64, 180))

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('visualize', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')


def main(argv):
    config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
    if FLAGS.train:
        # training
        # run_paths = utils_params.gen_run_folder(FLAGS.runId)
        # set loggers
        # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # utils_params.save_config(run_paths['path_gin'], gin.config_str())
        iscuda = common.torch_set_gpu(config["gpu"])
        common.mkdir_for(config["save_path"])

        # extract and create dataset from ROS Bag
        create_dataset = DatasetCreator(config)
        create_dataset()

        # dataset object for real pair lidar data
        real_pair_dt = RealPairDataset(root=config["data_dir"])
        # synth_pair_dt = SyntheticPairDataset(config["data_dir"], scale=RANDOM_SCALE,
        #                                   distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

        dataloader = PairLoader(dataset=real_pair_dt, scale=RANDOM_SCALE,
                                crop=RANDOM_CROP,
                                distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

        loader = threaded_loader(dataloader, iscuda=False, threads=1)

        print("\n>> Creating net = Quad_L2Net_ConfCFS")
        net = Quad_L2Net_ConfCFS()
        superpoint_net = SuperPoint(
            config={
                'descriptor_dim': 128,
                'max_keypoints': -1,
                'nms_radius': 4,
                'keypoint_threshold': 0.005
            })
        superglue_net = SuperGlue(
            config={
                'descriptor_dim': 128,
            }
        )
        print(f" ( Model size: {common.model_size(net) / 1000:.0f}K parameters )")

        sampler = NghSampler2(ngh=9, subd=1, pos_d=3, neg_d=5, border=10,
                              subd_neg=-8, maxpool_pos=True)

        for result in dataloader:
            aflow = torch.unsqueeze(torch.from_numpy(result["aflow"]), 0)
            org_flow = result["org_flow"]
            img1 = torch.unsqueeze(result["img1"], 0)
            img2 = torch.unsqueeze(result["img2"], 0)
            pred = {}
            data = {'image0': img1, 'image1': img2}
            out = net(imgs=[img1, img2])
            out1 = superpoint_net({'image': img1})
            out2 = superpoint_net({'image': img2})
            pred = {**pred, **{k + '0': v for k, v in out1.items()}}
            pred = {**pred, **{k + '1': v for k, v in out2.items()}}

            data = {**data, **pred}

            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])

            s_out = superglue_net(data)

            new_pred = {**pred, **s_out}

            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = new_pred['keypoints1'][0].cpu().numpy()
            matches0 = new_pred['matches0'][0].cpu().numpy()
            matches1 = new_pred['matches1'][0].cpu().numpy()
            confidence = new_pred['matching_scores0'][0].detach().numpy()

            pred_flow = np.full(org_flow.shape, -1, dtype=np.float32)
            pred_proj_x = kpts1[:, 0].astype(np.int)
            pred_proj_y = kpts1[:, 1].astype(np.int)
            pred_key_matches = kpts0[matches1]
            pred_flow[pred_proj_y, pred_proj_x, 0] = pred_key_matches[:, 1]
            pred_flow[pred_proj_y, pred_proj_x, 1] = pred_key_matches[:, 0]

            # sample = sampler(out.get("descriptors"), out.get("reliability"), aflow)
            print("res")
        #
        # loss = MultiLoss(
        #     # 1, ReliabilityLoss(sampler, base=0.5, nq=12),
        #     1, CosimLoss(N=8),
        #     1, PeakyLoss(N=8))
        #
        # # create optimizer
        # optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
        #                        lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))

        # train = MyTrainer(net, loader, loss, optimizer)
        loss = ReliabilityLoss(sampler, base=0.5, nq=12)

        train = TestTrainer(net, loader, loss)

        if iscuda:
            train = train.cuda()

        # Training loop #
        for epoch in range(config['epoch']):
            print(f"\n>> Starting epoch {epoch}...")
            train()

        print(f"\n>> Saving model to {config['save_path']}")
        torch.save({'net': 'Quad_L2Net_ConfCFS()', 'state_dict': net.state_dict()}, config["save_path"])

    else:
        # extract.test_model(config)
        matching.match_fun(config)

    if FLAGS.visualize:
        img, mask_array = random_image_loader(config["data_dir"])
        img_dict = dict(img=img, persp=(1, 0, 0, 0, 1, 0, 0, 0), mask=mask_array)
        synth_pair = SyntheticPairDataset(config["data_dir"], scale=RANDOM_SCALE,
                                          distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))
        img_scale, img_distort, metadata = synth_pair.get_pair(org_img=img_dict)
        loader = SythPairLoader(scale=RANDOM_RESCALE, distort=None, crop=RANDOM_CROP)
        result = loader.getitem(img_a=img_scale, img_b=img_distort, metadata=metadata)
        show_flow(img0=np.transpose(result['img1']), img1=np.transpose(result['img2']),
                  flow=np.transpose(result['aflow']), mask=np.transpose(result['mask']))
        print("done")


if __name__ == '__main__':
    app.run(main)
