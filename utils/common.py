# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb#, shutil
import numpy as np
import torch
import matplotlib.pyplot as plt


def mkdir_for(file_path):
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)


def model_size(model):
    ''' Computes the number of parameters of the model
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu>=0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'],os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True # speed-up cudnn
        torch.backends.cudnn.fastest = True # even more speed-up?
        # print( 'Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'] )

    else:
        print( 'Launching on CPU' )

    return cuda


def patch_extractor(image, patch_size=32, stride=32):
    patches = []
    for i in range(0, image.size(3)-patch_size+1, stride):
        patch = image[:, :, :, i:i+patch_size]
        patches.append(patch)
    return patches


def compare_images(in_img, out_img, out_channel=0):
    in_img = in_img.detach().squeeze().numpy()
    out_img = out_img.detach().squeeze().numpy()
    in_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())
    if out_img.ndim == 3:
        out_img = out_img[out_channel, :, :]
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(in_img)
    axes[1].imshow(out_img)
    plt.show()



