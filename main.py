import numpy as np
import random
from absl import app, flags
from input_pipeline.dataset_creator import DatasetCreator
from input_pipeline.dataset import RealPairDataset, SyntheticPairDataset, SingleDataset
from input_pipeline.dataloader import PairLoader, threaded_loader, SythPairLoader, SingleLoader
from input_pipeline.preprocessing import RandomScale, RandomTilting, PixelNoise, RandomTranslation, RandomCrop
import yaml
from utils import common
from models.model import SleceNet
from models.featurenet import AutoEncoder
import torch
from train import TrainAutoEncoder, TrainSleceNet
from evaluate import evaluate, test_network
import matplotlib.pyplot as plt

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
config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
iscuda = common.torch_set_gpu(config["gpu"])
device = torch.device("cuda" if iscuda else "cpu")


def main(argv):
    # run_paths = utils_params.gen_run_folder(FLAGS.runId)
    # set loggers
    # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # utils_params.save_config(run_paths['path_gin'], gin.config_str())
    common.mkdir_for(config["save_path"])

    # extract and create dataset from ROS Bag
    create_dataset = DatasetCreator(config, generate_ground_truth=config["gen_gt"])
    create_dataset()

    # synth_pair_dt = SyntheticPairDataset(config["data_dir"], scale=RANDOM_SCALE,
    #                                   distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

    print("\n>> Creating networks..")
    if config["train_fe"]:
        # dataset object for single lidar range images
        train_single_dt = SingleDataset(root=config["train_data_dir"])
        test_single_dt = SingleDataset(root=config["test_data_dir"])

        dataloader = SingleLoader(dataset=train_single_dt, scale=RANDOM_SCALE,
                                  crop=RANDOM_CROP, distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

        test_dataloader = SingleLoader(dataset=test_single_dt, scale=RANDOM_SCALE,
                                       crop=RANDOM_CROP, distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

        net = AutoEncoder().to(device)
        test_network("ae", dataloader, net)

        if FLAGS.train:
            loader = threaded_loader(dataloader, batch_size=4, iscuda=iscuda, threads=1)
            test_loader = threaded_loader(test_dataloader, batch_size=4, iscuda=iscuda, threads=1)
            train = TrainAutoEncoder(net=net, dataloader=loader, test_dataloader=test_loader,
                                     epochs=50, config=config, title="CAE_50", is_cuda=iscuda)
            train()

        else:
            evaluation(net, None)
    else:
        # dataset object for real pair lidar data
        train_r_pair_dt = RealPairDataset(root=config["train_data_dir"])
        test_r_pair_dt = RealPairDataset(root=config["test_data_dir"])

        dataloader = PairLoader(dataset=train_r_pair_dt, scale=RANDOM_SCALE,
                                crop=RANDOM_CROP,
                                distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))
        test_dataloader = PairLoader(dataset=test_r_pair_dt, scale=RANDOM_SCALE,
                                     crop=RANDOM_CROP,
                                     distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

        # net = FlowCorrelationCNN(config, device)
        net = SleceNet(config, device, iters=10).to(device)
        # test_network("s2lece", dataloader, net)

        if FLAGS.train:
            loader = threaded_loader(dataloader, batch_size=4, iscuda=iscuda, threads=1)
            test_loader = threaded_loader(test_dataloader, batch_size=4, iscuda=iscuda, threads=1)
            train = TrainSleceNet(net=net, dataloader=loader, test_dataloader=test_loader,
                                  epochs=10, config=config, is_cuda=iscuda)
            train()

        else:
            # random_evaluation(net)
            evaluation(net, test_r_pair_dt)


def evaluation(net, dataset):
    i = random.randint(0, dataset.npairs)
    img_a, img_b, metadata = dataset.get_pair(i)
    aflow = np.float32(metadata.pop('aflow'))
    mask = metadata.pop('flow_mask', np.ones(aflow.shape[:2], np.uint8))
    net_weights = torch.load(config['save_path'])
    net.load_state_dict(net_weights["state_dict"])
    net.eval()
    evaluate(net, img_a, img_b, aflow, mask, i, metadata)


def random_evaluation(net):
    def load(id1, id2):
        img1 = np.load(f'dataset/data/ae_val_data/{id1}/range.npy')
        idx1 = np.load(f'dataset/data/ae_val_data/{id1}/idx.npy')
        xyz1 = np.load(f'dataset/data/ae_val_data/{id1}/xyz.npy')
        img2 = np.load(f'dataset/data/ae_val_data/{id2}/range.npy')
        idx2 = np.load(f'dataset/data/ae_val_data/{id2}/idx.npy')
        xyz2 = np.load(f'dataset/data/ae_val_data/{id2}/xyz.npy')
        return img1, img2, {'idx1': idx1, 'idx2': idx2, 'xyz1': xyz1, 'xyz2': xyz2 }
    img_a, img_b, metadata = load(10, 20)
    net_weights = torch.load(config['save_path'])
    net.load_state_dict(net_weights["state_dict"])
    net.eval()
    evaluate(net, img_a, img_b, metadata=metadata, random=True)


if __name__ == '__main__':
    app.run(main)
