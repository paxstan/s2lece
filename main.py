import numpy as np
import logging
import random
from absl import app, flags
from input_pipeline.dataset_creator import DatasetCreator
from input_pipeline.dataset import RealPairDataset, SingleDataset
from input_pipeline.dataloader import PairLoader, threaded_loader, SingleLoader
import yaml
from utils import common, utils_params, utils_misc
from models.model import SleceNet
from models.featurenet import AutoEncoder
import torch
from train import TrainAutoEncoder, TrainSleceNet
from evaluate import evaluate, test_network
from tune import TuneS2leceNet
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from input_pipeline.preprocessing import PixelNoise, RandomTilting

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('tune', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('visualize', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')
config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
iscuda = common.torch_set_gpu(config["gpu"])
device = torch.device("cuda" if iscuda else "cpu")


# RANDOM_NOISE = PixelNoise(ampl=50)
# RANDOM_TILT = RandomTilting(magnitude=0.025, directions="left")


def main(argv):
    logging.info("Starting the script......")
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # extract and create dataset from ROS Bag
    create_dataset = DatasetCreator(config)
    data_dir = create_dataset()

    fe_params = config["autoencoder"]["params"]
    sl_params = config["s2lece"]["params"]

    if FLAGS.train:
        print("\n>> Creating networks..")
        if config["train_fe"]:
            # dataset object for single lidar range images
            train_single_dt = SingleDataset(root=data_dir)
            # test_single_dt = SingleDataset(root=config["autoencoder"]["test_data_dir"])

            dataloader = SingleLoader(dataset=train_single_dt)

            train_loader, val_loader = train_test_split(dataloader, train_size=0.7, random_state=42)
            val_loader, test_loader = train_test_split(val_loader, train_size=0.5, random_state=42)

            # params = config["autoencoder"]["params"]
            net = AutoEncoder(fe_params).to(device)

            if FLAGS.train:
                train_thread_loader = threaded_loader(train_loader, batch_size=4, iscuda=iscuda, threads=1)
                val_thread_loader = threaded_loader(val_loader, batch_size=4, iscuda=iscuda, threads=1)
                train = TrainAutoEncoder(net=net, train_loader=train_thread_loader, val_loader=val_thread_loader,
                                         config=config, title="ResNet", is_cuda=iscuda,
                                         max_count=len(dataloader), run_paths=run_paths)
                train()

            else:
                # test_network("ae", dataloader, net)
                evaluation(net, test_loader)
        else:
            # dataset object for real pair lidar data
            train_r_pair_dt = RealPairDataset(root=data_dir)
            # test_r_pair_dt = RealPairDataset(root=config["test_data_dir"])

            # dataloader = PairLoader(dataset=train_r_pair_dt)
            # test_dataloader = PairLoader(dataset=test_r_pair_dt)
            train_size = int(0.8 * len(train_r_pair_dt))  # 80% for training
            test_size = len(train_r_pair_dt) - train_size

            train_dataset, test_dataset = random_split(train_r_pair_dt, [train_size, test_size])

            # train_loader, val_loader = train_test_split(dataloader, train_size=0.7, random_state=42)
            # val_loader, test_loader = train_test_split(val_loader, train_size=0.5, random_state=42)

            net = SleceNet(config, device, sl_params, fe_params).to(device)
            test_network("s2lece", train_dataset, net)

            if FLAGS.train:
                loader = threaded_loader(train_dataset, batch_size=4, iscuda=iscuda, threads=1)
                test_loader = threaded_loader(test_dataset, batch_size=4, iscuda=iscuda, threads=1)
                train = TrainSleceNet(net=net, dataloader=loader, test_dataloader=test_loader, config=config,
                                      run_paths=run_paths, is_cuda=iscuda)
                train()

            elif FLAGS.tune:
                tune = TuneS2leceNet(config, dataloader, val_loader, run_paths, iscuda, device)
                tune()

            else:
                # random_evaluation(net)
                evaluation(net, test_r_pair_dt)


def evaluation(net, dataset):
    i = random.randint(0, dataset.npairs)
    img_a, img_b, metadata = dataset.get_pair(i)
    aflow = np.float32(metadata.pop('aflow'))
    mask2 = metadata.get('mask2', np.ones(aflow.shape[:2], np.uint8))
    net_weights = torch.load(config['save_path'])
    net.load_state_dict(net_weights["model_state_dict"])
    net.eval()
    evaluate(net, img_a, img_b, aflow, mask2, i, metadata)


def random_evaluation(net):
    def load(id1, id2):
        img1 = np.load(f'dataset/data/ae_val_data/{id1}/range.npy')
        idx1 = np.load(f'dataset/data/ae_val_data/{id1}/idx.npy')
        xyz1 = np.load(f'dataset/data/ae_val_data/{id1}/xyz.npy')
        img2 = np.load(f'dataset/data/ae_val_data/{id2}/range.npy')
        idx2 = np.load(f'dataset/data/ae_val_data/{id2}/idx.npy')
        xyz2 = np.load(f'dataset/data/ae_val_data/{id2}/xyz.npy')
        mask2 = np.load(f'dataset/data/ae_val_data/{id2}/valid_mask.npy')
        return img1, img2, {'idx1': idx1, 'idx2': idx2, 'xyz1': xyz1, 'xyz2': xyz2, 'mask2': mask2}

    img_a, img_b, metadata = load(10, 1000)
    mask_b = metadata.get('mask2')
    net_weights = torch.load(config['save_path'])
    net.load_state_dict(net_weights["model_state_dict"])
    net.eval()
    evaluate(net, img_a, img_b, valid_mask=mask_b, metadata=metadata, random=True)


if __name__ == '__main__':
    app.run(main)
