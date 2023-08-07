import numpy as np
import os
import logging
import random
from absl import app, flags
from input_pipeline.dataset_creator import DatasetCreator
from input_pipeline.dataset import RealPairDataset, SingleDataset
from input_pipeline.dataloader import threaded_loader
import yaml
from utils import utils_params, utils_misc
from models.model import S2leceNet
from models.featurenet import AutoEncoder, FeatureExtractorNet
import torch
from train import TrainAutoEncoder, TrainSleceNet
from evaluate import evaluate, test_network
from tune import TuneS2leceNet
from torch.utils.data import random_split, ConcatDataset
from models.model_utils import load_encoder_state_dict

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('tune', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('visualize', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')
config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
iscuda = utils_misc.torch_set_gpu(config["gpu"])
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
            #     # dataset object for single lidar range images
            #     combined_dataset = []
            #     for dt in config["datasets"]:
            #         dataset = config["dataset"][dt]
            #         if os.path.exists(dataset['data_dir']):
            #             single_dt = SingleDataset(root=dataset['data_dir'])
            #             combined_dataset.append(single_dt)
            #
            #     full_dataset = ConcatDataset(combined_dataset)

            single_dt = SingleDataset(root=data_dir)
            # val_single_dt = SingleDataset(root="../dataset/exp04")

            train_size = int(0.8 * len(single_dt))  # 80% for training
            val_size = len(single_dt) - train_size

            train_dataset, val_dataset = random_split(single_dt, [train_size, val_size])

            # meanr = 0.0
            # stdr = 0.0
            # meanx = 0.0
            # stdx = 0.0
            # meany = 0.0
            # stdy = 0.0
            # meanz = 0.0
            # stdz = 0.0
            # for idx, inputs in enumerate(train_dataset):
            #     img = inputs.pop("un_proj_img")
            #     unproj_xyz = inputs.pop("un_proj_xyz")
            #     npoints = unproj_xyz.shape[0]
            #     meanr += img[:npoints].mean()
            #     stdr += img[:npoints].std()
            #     meanx += unproj_xyz[:npoints, 0].mean()
            #     stdx += unproj_xyz[:npoints, 0].std()
            #     meany += unproj_xyz[:npoints, 1].mean()
            #     stdy += unproj_xyz[:npoints, 1].std()
            #     meanz += unproj_xyz[:npoints, 2].mean()
            #     stdz += unproj_xyz[:npoints, 2].std()
            #
            # meanr /= len(train_dataset)
            # stdr /= len(train_dataset)
            # meanx /= len(train_dataset)
            # stdx /= len(train_dataset)
            # meany /= len(train_dataset)
            # stdy /= len(train_dataset)
            # meanz /= len(train_dataset)
            # stdz /= len(train_dataset)

            # for idx, (img, means, stds) in enumerate(val_dataset):
            #     print(means, std)
            #     mean += means
            #     std += stds

            # fe_params["in_channel"] = 4
            net = AutoEncoder(fe_params).to(device)
            test_network("ae", train_dataset, net)

            train_loader = threaded_loader(train_dataset, batch_size=2, iscuda=iscuda, threads=1)
            val_loader = threaded_loader(val_dataset, batch_size=2, iscuda=iscuda, threads=1, shuffle=False)

            if FLAGS.train:
                train = TrainAutoEncoder(net=net, train_loader=train_loader, val_loader=val_loader,
                                         config=config, title="DarkNet", is_cuda=iscuda, run_paths=run_paths)
                train()

            elif FLAGS.tune:
                tune = TuneS2leceNet(config, train_loader, val_loader, run_paths, iscuda, device)
                tune()

            else:
                evaluation(net, val_loader)
        else:
            # dataset object for real pair lidar data
            pair_dt = RealPairDataset(root=data_dir)

            train_size = int(0.8 * len(pair_dt))  # 80% for training
            val_size = len(pair_dt) - train_size

            train_dataset, val_dataset = random_split(pair_dt, [train_size, val_size])

            # val_size = int(0.8 * len(val_dataset))
            # test_size = len(val_dataset) - val_size
            # val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

            # train_loader, val_loader = train_test_split(dataloader, train_size=0.7, random_state=42)
            # val_loader, test_loader = train_test_split(val_loader, train_size=0.5, random_state=42)

            feature_net = AutoEncoder(fe_params).to(device)
            encoder_state_dict = load_encoder_state_dict(feature_net, config["autoencoder"]["save_path"])
            net = S2leceNet(config, fe_params, sl_params)
            net.load_encoder(encoder_state_dict)
            net.to(device)
            test_network("s2lece", val_dataset, net)

            train_loader = threaded_loader(train_dataset, batch_size=config["s2lece"]["batch_size"],
                                           iscuda=iscuda, threads=1)
            val_loader = threaded_loader(train_dataset, batch_size=config["s2lece"]["batch_size"],
                                         iscuda=iscuda, threads=1, shuffle=False)

            if FLAGS.train:
                train = TrainSleceNet(net=net, dataloader=train_loader, test_dataloader=val_loader, config=config,
                                      run_paths=run_paths, device=device, is_cuda=iscuda)
                train()

            elif FLAGS.tune:
                tune = TuneS2leceNet(config, train_loader, val_loader, run_paths, iscuda, device)
                tune()

            else:
                # random_evaluation(net)
                evaluation(net, val_dataset)


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
