import numpy as np
from absl import app, flags
from input_pipeline.dataset import random_image_loader, DatasetCreator
from input_pipeline.dataset_classes import SyntheticPair, LidarPairDataset
from input_pipeline.dataloader import PairLoader, threaded_loader, SythPairLoader
from input_pipeline.preprocessing import RandomScale, RandomTilting, PixelNoise, RandomTranslation, RandomCrop
import yaml
from visualization.visualization import show_flow
from utils import common
from utils.train import MyTrainer
from models.patchnet import Quad_L2Net_ConfCFS
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
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
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
        create_dataset = DatasetCreator(config)
        create_dataset()
        lidar_pair_dt = LidarPairDataset(root=config["data_dir"])
        dataloader = PairLoader(dataset=lidar_pair_dt, scale=RANDOM_SCALE,
                                crop=RANDOM_CROP,
                                distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))
        loader = threaded_loader(dataloader, iscuda=False, threads=1)

        print("\n>> Creating net = Quad_L2Net_ConfCFS")
        net = Quad_L2Net_ConfCFS()
        print(f" ( Model size: {common.model_size(net) / 1000:.0f}K parameters )")

        sampler = NghSampler2(ngh=9, subq=-8, subd=1, pos_d=3, neg_d=5, border=10,
                              subd_neg=-8, maxpool_pos=True)

        loss = MultiLoss(
            1, ReliabilityLoss(sampler, base=0.5, nq=12),
            1, CosimLoss(N=8),
            1, PeakyLoss(N=8))

        # create optimizer
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                               lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))

        train = MyTrainer(net, loader, loss, optimizer)

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
        synth_pair = SyntheticPair(config["data_dir"], scale=RANDOM_SCALE,
                                   distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))
        img_scale, img_distort, metadata = synth_pair.get_pair(org_img=img_dict)
        loader = SythPairLoader(scale=RANDOM_RESCALE, distort=None, crop=RANDOM_CROP)
        result = loader.getitem(img_a=img_scale, img_b=img_distort, metadata=metadata)
        show_flow(img0=np.transpose(result['img1']), img1=np.transpose(result['img2']),
                  flow=np.transpose(result['aflow']), mask=np.transpose(result['mask']))
        print("done")


if __name__ == '__main__':
    app.run(main)
