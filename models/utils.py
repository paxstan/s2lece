import torch.nn as nn
import torch.nn.functional as F
import torch
from spatial_correlation_sampler import spatial_correlation_sample
import math


def conv_layer(name, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), max_pooling=True):
    seq_model = nn.Sequential()
    seq_model.add_module(f"conv_{name}",
                         nn.Conv2d(in_channel, out_channel,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2), bias=False))
    seq_model.add_module(f"bn_{name}", nn.BatchNorm2d(out_channel))
    seq_model.add_module(f"activ_{name}", nn.LeakyReLU(0.1, inplace=True))
    if max_pooling:
        seq_model.add_module(f"max_pool_{name}", nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
    return seq_model


def de_conv_layer(name, in_channel, out_channel, kernel_size=(3, 3), max_pooled=True, decoder=False):
    seq_model = nn.Sequential()
    if max_pooled:
        seq_model.add_module(f"de_conv_{name}",
                             nn.ConvTranspose2d(in_channel, out_channel,
                                                kernel_size=kernel_size, stride=(1, 2),
                                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                                output_padding=(0, 1),
                                                bias=False))
    else:
        seq_model.add_module(f"de_conv_{name}",
                             nn.ConvTranspose2d(in_channel, out_channel,
                                                kernel_size=kernel_size, stride=1,
                                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                                bias=False))
    if decoder:
        seq_model.add_module(f"de_bn_{name}", nn.BatchNorm2d(out_channel))
        seq_model.add_module(f"de_activ_{name}", nn.LeakyReLU(0.1, inplace=True))
    return seq_model


def predict_flow_layer(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def correlate(input1, input2, patch_size=4, dilation_patch=2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=patch_size,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=dilation_patch)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    # out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
    # visualize_correlation(out_corr, ph)
    out_corr = out_corr / input1.size(1)
    return F.leaky_relu(out_corr, 0.01)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    # mask = torch.autograd.Variable(torch.ones(x.size()))

    if x.is_cuda:
        grid = grid.cuda()
        # mask = mask.cuda()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)

    mask = nn.functional.grid_sample(torch.ones_like(x), vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def up_sample_flow(flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8 * H, 8 * W)


def linear_position_embedding_sine(x, dim=128, NORMALIZE_FACOR=1 / 200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    b, dim, f = x.size()
    frequency = torch.linspace(0, dim, dim).to(x.device)
    pos_tensor = torch.ones_like(x)
    for i in range(dim):
        if i % 2 == 0:
            pos_tensor[:, i] = torch.sin(x[:, i] * (i + 1) * pos_tensor[:, i] * math.pi)
        else:
            pos_tensor[:, i] = torch.cos(x[:, i] * (i + 1) * pos_tensor[:, i] * math.pi)

    # pos_em1 = torch.sin(math.pi * x[..., -2:-1] * frequency) + torch.cos(
    #     math.pi * x[..., -2:-1] * frequency)
    # pose_em2 = torch.sin(math.pi * x[..., -1:] * frequency) + torch.cos(math.pi * x[..., -1:] * frequency)
    # pos_embed = torch.cat([
    #     torch.sin(frequency * math.pi * x[..., -2:-1]) + torch.cos(frequency * math.pi * x[..., -2:-1]),
    #     torch.sin(frequency * math.pi * x[..., -1:]) + torch.cos(frequency * math.pi * x[..., -1:])
    # ], dim=-1)
    return pos_tensor
    # freq_bands = torch.linspace(0, dim, dim).to(x.device)  # (0, 127, 128)
    # return torch.cat([
    #     torch.sin(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
    #     torch.cos(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
    #     torch.sin(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR),
    #     torch.cos(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR)],
    #     dim=-1)


def exp_position_embedding_sine(x, dim=128, NORMALIZE_FACOR=1 / 200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    freq_bands = torch.linspace(0, dim // 4 - 1, dim // 4).to(x.device)
    return torch.cat([torch.sin(x[..., -2:-1] * (NORMALIZE_FACOR * 2 ** freq_bands)),
                      torch.cos(x[..., -2:-1] * (NORMALIZE_FACOR * 2 ** freq_bands)),
                      torch.sin(x[..., -1:] * (NORMALIZE_FACOR * 2 ** freq_bands)),
                      torch.cos(x[..., -1:] * (NORMALIZE_FACOR * 2 ** freq_bands))], dim=-1)


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=512, sine_type='lin_sine'):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i + 1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2 ** i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat(
            [torch.sin(i * math.pi * inputs) for i in self.bases] + [torch.cos(i * math.pi * inputs) for i in
                                                                     self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out
