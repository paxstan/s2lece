import numpy as np
from absl import app, flags
from input_pipeline.dataset import random_image_loader, DatasetCreator
from input_pipeline.dataset_classes import SyntheticPair
from input_pipeline.dataloader import PairLoader
from input_pipeline.preprocessing import RandomScale, RandomTilting, PixelNoise, RandomTranslation, ColorJitter, \
    RandomCrop
import yaml
from visualization.visualization import show_flow
from utils import utils_params

RANDOM_SCALE = RandomScale(min_size=80, max_size=128, can_upscale=True)
RANDOM_TILT = RandomTilting(magnitude=0.025, directions="left")
RANDOM_NOISE = PixelNoise(ampl=50)
RANDOM_TRANS = RandomTranslation(roll=100)
RANDOM_RESCALE = RandomScale(64, 64, can_upscale=True)
RANDOM_JITTER = ColorJitter(0.1, 0.1, 0.2, 0.1)
RANDOM_CROP = RandomCrop((64, 180))

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')


def main(argv):
    config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
    if FLAGS.train:
        # training
        # run_paths = utils_params.gen_run_folder(FLAGS.runId)
        # set loggers
        # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # utils_params.save_config(run_paths['path_gin'], gin.config_str())
        dataset = DatasetCreator(config)
        dataset()
    else:
        img, mask_array = random_image_loader(config["data_dir"])
        img_dict = dict(img=img, persp=(1, 0, 0, 0, 1, 0, 0, 0), mask=mask_array)
        synth_pair = SyntheticPair(scale=RANDOM_SCALE,
                                   distort=(RANDOM_TILT, RANDOM_NOISE, RANDOM_TRANS))
        img_scale, img_distort, metadata = synth_pair.get_pair(org_img=img_dict)
        loader = PairLoader(scale=RANDOM_RESCALE, distort=None, crop=RANDOM_CROP)
        result = loader.getitem(img_a=img_scale, img_b=img_distort, metadata=metadata)
        show_flow(img0=np.transpose(result['img1']), img1=np.transpose(result['img2']),
                  flow=np.transpose(result['aflow']), mask=np.transpose(result['mask']))
        print("done")


if __name__ == '__main__':
    app.run(main)
