import numpy as np
import random
from absl import app, flags
from input_pipeline.dataset_creator import random_image_loader, DatasetCreator
from input_pipeline.dataset import RealPairDataset, SyntheticPairDataset, SingleDataset
from input_pipeline.dataloader import PairLoader, threaded_loader, SythPairLoader, SingleLoader
from input_pipeline.preprocessing import RandomScale, RandomTilting, PixelNoise, RandomTranslation, RandomCrop
import yaml
from visualization.visualization import show_flow
from utils import common
from models.model import FlowModel
import torch
from train import train
from evaluate import evaluate

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

    # run_paths = utils_params.gen_run_folder(FLAGS.runId)
    # set loggers
    # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # utils_params.save_config(run_paths['path_gin'], gin.config_str())
    iscuda = common.torch_set_gpu(config["gpu"])
    device = torch.device("cuda" if iscuda else "cpu")
    common.mkdir_for(config["save_path"])

    # extract and create dataset from ROS Bag
    create_dataset = DatasetCreator(config)
    create_dataset()

    # dataset object for single lidar range images
    # single_dt = SingleDataset(root=config["data_dir"])
    # single_dataloader = SingleLoader(dataset=single_dt, scale=RANDOM_SCALE,
    #                                  crop=RANDOM_CROP, distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

    # dataset object for real pair lidar data
    real_pair_dt = RealPairDataset(root=config["data_dir"])
    # synth_pair_dt = SyntheticPairDataset(config["data_dir"], scale=RANDOM_SCALE,
    #                                   distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

    net = FlowModel(device)

    if FLAGS.train:
        # training
        dataloader = PairLoader(dataset=real_pair_dt, scale=RANDOM_SCALE,
                                crop=RANDOM_CROP,
                                distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))

        loader = threaded_loader(dataloader, batch_size=4, iscuda=iscuda, threads=1)

        print("\n>> Creating networks..")

        # feature_net = FeatureExtractor().to(device)
        # correlation_net = CorrelationNetwork().to(device)

        for input_data in dataloader:
            img1 = input_data.pop('img1')
            img2 = input_data.pop('img2')
            flow = input_data.pop('aflow')
            img1 = torch.unsqueeze(torch.tensor(img1), 0)
            img2 = torch.unsqueeze(torch.tensor(img2), 0)
            target_flow = torch.unsqueeze(torch.tensor(flow), 0)
            pred_flow = net(img1, img2)
        #     epe_loss = torch.norm(target_flow-pred_flow, p=2, dim=1)
        #     flow_mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        #     epe_loss = epe_loss[~flow_mask]
        #     # new_flow = torch.concat((img2, flow), dim=1)
        #     # out1 = cnn_autoencoder(img1)
        #     # out2 = cnn_autoencoder(img2)
        #     # output = correlation_net(out1, out2, flow, new_flow)
        #     print(f"loss :{epe_loss}")

        train(net, dataloader=loader, epochs=10, config=config)

    else:
        i = random.randint(0, real_pair_dt.npairs)
        img_a, img_b, metadata = real_pair_dt.get_pair(i)
        aflow = np.float32(metadata['aflow'])
        net_weights = torch.load(config['save_path'])
        net.load_state_dict(net_weights["state_dict"])
        net.eval()
        evaluate(net, img_a, img_b, aflow)

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
