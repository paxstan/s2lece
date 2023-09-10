import logging
from absl import app, flags
from input_pipeline.dataset_creator import DatasetCreator, SyntheticDatasetCreator, KittiDatasetCreator
from input_pipeline.dataset import RealPairDataset, SingleDataset
from input_pipeline.dataloader import threaded_loader
import yaml
from utils import utils_params, utils_misc
from models.model import S2leceNet
from models.featurenet import AutoEncoder
import torch
from train import TrainAutoEncoder, TrainS2leceNet
from evaluate import evaluate, test_network
from torch.utils.data import ConcatDataset
from utils.utils_misc import load_encoder_state_dict, random_split_dataset

# necessary flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('model', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
# flags.DEFINE_boolean('visualize', False, 'Specify whether to train or evaluate a model.')
# flags.DEFINE_string('runId', "/home/paxstan/Documents/research_project/code/runs/1_run_final_report_kitti",
#                     'Specify path to the run directory.')
flags.DEFINE_string('runId', "",
                    'Specify path to the run directory.')
config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
iscuda = utils_misc.torch_set_gpu(config["gpu"])
device = torch.device("cuda" if iscuda else "cpu")


def main(argv):
    logging.info("Starting the script......")

    # create run folder
    run_paths = utils_params.gen_run_folder(FLAGS.runId)
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # extract from HITLI Dataset
    # create_dataset = DatasetCreator(config)
    # data_dir = create_dataset()

    # create_synth_dataset = SyntheticDatasetCreator(config)
    # data_dir = create_synth_dataset()

    # extract from KITTI Dataset
    train_dataset = KittiDatasetCreator(config, dataset_name="kitti_03")
    train_data = train_dataset()

    val_dataset = KittiDatasetCreator(config, dataset_name="kitti_04")
    val_data = val_dataset()

    synth_np_dataset = KittiDatasetCreator(config, dataset_name="synth_np_kitti_03", synth=True)
    synth_np_data = synth_np_dataset()

    synth_np_dataset2 = KittiDatasetCreator(config, dataset_name="synth_np_kitti_04", synth=True)
    synth_np_data2 = synth_np_dataset2()

    synth_dataset = KittiDatasetCreator(config, dataset_name="synth_kitti_03", synth=True)
    synth_data = synth_dataset()

    synth_dataset2 = KittiDatasetCreator(config, dataset_name="synth_kitti_04", synth=True)
    synth_data2 = synth_dataset2()

    # parameter for model
    fe_params = config["autoencoder"]["params"]
    sl_params = config["s2lece"]["params"]

    if FLAGS.model:
        print("\n>> Creating networks..")
        if config["train_fe"]:
            # train / evaluate autoencoder
            dataset1 = SingleDataset(dataset=train_data)
            dataset2 = SingleDataset(dataset=val_data)
            dataset3 = SingleDataset(dataset=synth_data)
            dataset4 = SingleDataset(dataset=synth_data2)
            dataset5 = SingleDataset(dataset=synth_np_data)
            dataset6 = SingleDataset(dataset=synth_np_data2)

            # combine datasets
            train_dataset = ConcatDataset([dataset1, dataset3, dataset4, dataset5])
            val_dataset = ConcatDataset([dataset2, dataset6])

            net = AutoEncoder(fe_params).to(device)
            test_network("ae", train_dataset, net)

            if FLAGS.train:
                # train autoencoder
                train_loader = threaded_loader(train_dataset, batch_size=config["autoencoder"]["batch_size"],
                                               iscuda=iscuda, threads=1)
                val_loader = threaded_loader(val_dataset, batch_size=config["autoencoder"]["batch_size"],
                                             iscuda=iscuda, threads=1, shuffle=False)
                train = TrainAutoEncoder(net=net, train_loader=train_loader, val_loader=val_loader,
                                         config=config, title="Autoencoder", is_cuda=iscuda, run_paths=run_paths)
                train()
            else:
                # evaluate autoencoder
                net_weights = torch.load(config['autoencoder']['pretrained_path'])
                net.load_state_dict(net_weights["state_dict"])
                net.eval()
                inputs = next(iter(val_dataset))
                evaluate(net, "ae", inputs, run_paths)
        else:
            # train / evaluate s2lece
            dataset1 = RealPairDataset(dataset=train_data)
            dataset2 = RealPairDataset(dataset=val_data)

            train_dataset1, val_dataset1 = random_split_dataset(dataset1, 0.9)
            train_dataset2, val_dataset2 = random_split_dataset(dataset2, 0.9)

            train_dataset = ConcatDataset([train_dataset1, train_dataset2])
            val_dataset = ConcatDataset([val_dataset1, val_dataset2])

            net = S2leceNet(config["s2lece"]["iters"], fe_params, sl_params).to(device)
            # test_network("s2lece", train_dataset, net)

            if FLAGS.train:
                # train s2lece
                train_loader = threaded_loader(train_dataset, batch_size=config["s2lece"]["batch_size"],
                                               iscuda=iscuda, threads=1)
                val_loader = threaded_loader(val_dataset, batch_size=config["s2lece"]["batch_size"],
                                             iscuda=iscuda, threads=1, shuffle=False)
                feature_net = AutoEncoder(fe_params).to(device)
                encoder_state_dict = load_encoder_state_dict(feature_net, config["s2lece"]["ae_path"])
                net = S2leceNet(config["s2lece"]["iters"], fe_params, sl_params)

                # load encoder weights from Autoencoder
                net.load_encoder(encoder_state_dict)
                train = TrainS2leceNet(net=net, dataloader=train_loader, test_dataloader=val_loader, config=config,
                                      run_paths=run_paths, device=device, is_cuda=iscuda)
                train()

            else:
                # evaluate s2lece
                # random_evaluation(net)
                net_weights = torch.load(config['s2lece']['pretrained_path'])
                net.load_state_dict(net_weights["state_dict"])
                net.eval()
                for i in range(len(train_dataset)):
                    inputs = next(iter(train_dataset))
                    evaluate(net, "slece", inputs, run_paths)


if __name__ == '__main__':
    app.run(main)
