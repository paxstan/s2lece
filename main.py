import numpy as np
from absl import app
from input_pipeline.dataset import SyntheticPair
from input_pipeline.dataloader import PairLoader
from input_pipeline.preprocessing import RandomScale, RandomTilting, PixelNoise, RandomTranslation, ColorJitter, \
    RandomCrop
import yaml
from visualization.visualization import show_flow
# from utils import utils_params
# import numpy
from PIL import Image


def main(argv):
    # generate folder structures
    # run_paths = utils_params.gen_run_folder(FLAGS.runId)
    # # set loggers
    # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    #
    # utils_params.save_config(run_paths['path_gin'], gin.config_str())
    config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
    # dataset.point_cloud_extractor(path=config['ros_bag_path'], data_dir=config['data_dir'])
    # dataset.generate_data(data_dir=config['data_dir'])
    img_array = np.load('/home/paxstan/Documents/Uni/research_project/data/np_data/1358/range.npy')
    mask_array = np.load('/home/paxstan/Documents/Uni/research_project/data/np_data/1358/valid_mask.npy')
    im = ((img_array / 8. + 1.) / 2. * 255).astype(np.uint8)
    img = Image.fromarray(im)
    img_dict = dict(img=img, persp=(1, 0, 0, 0, 1, 0, 0, 0), mask=mask_array)
    random_scale = RandomScale(min_size=80, max_size=128, can_upscale=True)
    random_tilt = RandomTilting(magnitude=0.025, directions="left")
    random_noise = PixelNoise(ampl=50)
    random_trans = RandomTranslation(roll=100)
    synth_pair = SyntheticPair(scale=random_scale, distort=(random_tilt, random_noise, random_trans))
    img_scale, img_distort, metadata = synth_pair.get_pair(org_img=img_dict)
    random_rescale = RandomScale(64, 64, can_upscale=True)
    random_jitter = ColorJitter(0.1, 0.1, 0.2, 0.1)
    random_crop = RandomCrop((64, 180))
    loader = PairLoader(scale=random_rescale, distort=None, crop=random_crop)
    result = loader.getitem(img_a=img_scale, img_b=img_distort, metadata=metadata)
    show_flow(img0=np.transpose(result['img1']), img1=np.transpose(result['img2']),
              flow=np.transpose(result['aflow']), mask=result['mask'])
    print(result)


if __name__ == '__main__':
    app.run(main)
