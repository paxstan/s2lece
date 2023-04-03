from absl import app, flags
from input_pipeline import dataset
import yaml
from utils import utils_params

def main(argv):
    # generate folder structures
    # run_paths = utils_params.gen_run_folder(FLAGS.runId)
    # # set loggers
    # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    #
    # utils_params.save_config(run_paths['path_gin'], gin.config_str())
    config = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)
    # dataset.point_cloud_extractor(path=config['ros_bag_path'], data_dir=config['data_dir'])
    dataset.generate_data(data_dir=config['data_dir'])


if __name__ == '__main__':
    app.run(main)
