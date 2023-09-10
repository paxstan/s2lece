import logging
import torch
import os
from torch.utils.data import random_split


def set_loggers(path_log=None, logging_level=0, b_stream=False):
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        print('Launching on CPU')

    return cuda


def load_encoder_state_dict(feature_net, ae_path):
    if os.path.exists(ae_path):
        fe_net_weights = torch.load(ae_path)
        feature_net.load_state_dict(fe_net_weights["state_dict"])
        print(f"AE Model loaded from {ae_path}")
    else:
        print(f"AE Model is not in the path {ae_path}")
    return feature_net.encoder.state_dict()


def random_split_dataset(dataset, percent):
    train_size = int(percent * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
