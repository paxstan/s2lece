import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spatial_correlation_sampler import spatial_correlation_sample
from visualization.visualization import flow_to_color


class FlowModel(nn.Module):
    def __init__(self, config, device, train=True):
        super(FlowModel, self).__init__()
        self.config = config
        self.device = device
        self.to(self.device)
        self.feature_net = FeatureExtractorNet().to(self.device)
        self.correlation_net = CorrelationNet().to(self.device)
        if train:
            self.load_encoder()
        self.patch_size = [4, 8, 8, 16, 16, 32]
        self.dilation_patch = [2, 4, 4, 8, 8, 16]

    def load_encoder(self):
        if os.path.exists(self.config['fe_save_path']):
            fe_net_weights = torch.load(self.config['fe_save_path'])
            self.feature_net.load_state_dict(fe_net_weights["state_dict"])
            for params in self.feature_net.parameters():
                params.requires_grad = False
            print(f"AE Model loaded from {self.config['fe_save_path']}")
        else:
            print(f"AE Model is not in the path {self.config['fe_save_path']}")

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, x1, x2):
        predict_flow = None
        x1_fe_out = [x1]
        x2_fe_out = [x2]

        for layers in self.feature_net.encoder:
            x1 = layers(x1)
            x2 = layers(x2)
            x1_fe_out.append(x1)
            x2_fe_out.append(x2)

        i = len(x1_fe_out)-1
        j = 0
        wrapped_x2_out = x2_fe_out[i]

        for layers in self.correlation_net.correlation:
            x1_x2_correlate = correlate(x1_fe_out[i], wrapped_x2_out,
                                        patch_size=self.patch_size[j],
                                        dilation_patch=self.dilation_patch[j])  # [1, 16, 4, 128]
            x1_corr_concat = torch.cat([x1_fe_out[i], x1_x2_correlate], dim=1)  # [1, 272, 4, 128]

            predict_flow = layers(x1_corr_concat)
            # flow_img_4 = flow_to_color(self.predict_flow.detach().squeeze().numpy().transpose(1, 2, 0))
            if i != 0:
                wrapped_x2_out = warp(x2_fe_out[i - 1], predict_flow)
                i = i-1
            j = j + 1
        return predict_flow


class AutoEncoder(nn.Module):
    def __init__(self, in_channel=1, latent_dim=30, vae=False):
        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.vae = vae
        self.encoder = FeatureExtractorNet(in_channel)
        if vae:
            self.flatten = nn.Flatten()
            self.linear_mu = nn.Sequential(
                nn.Linear(256 * 32 * 128, latent_dim),
                nn.Tanh()
            )
            self.linear_logvar = nn.Sequential(
                nn.Linear(256 * 32 * 128, latent_dim),
                nn.Tanh()
            )
            self.reverse_linear = nn.Linear(latent_dim, 256 * 32 * 128)
            # self.conv1 = nn.Conv2d(256, latent_dim, 16, stride=1, padding=0)
            # self.conv2 = nn.Conv2d(256, latent_dim, 16, stride=1, padding=0)
            # self.conv3 = nn.ConvTranspose2d(latent_dim, 256, 1, stride=1, padding=0)
        self.decoder = Decoder(in_channel)

    def bottleneck(self, x):
        # mu, log_var = self.conv1(x), self.conv2(x)
        x = x.view(x.size(0), -1)
        mu, log_var = self.linear_mu(x), self.linear_logvar(x)
        z = self.re_parameterize(mu, log_var)
        return z, mu, log_var

    @staticmethod
    def re_parameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z

    def forward(self, x):
        mu = 0
        log_var = 0
        for layers in self.encoder.encoder:
            x = layers(x)
        if self.vae:
            x, mu, log_var = self.bottleneck(x)
            x = self.reverse_linear(x)
            x = x.view(x.size(0), 256, 32, 128)
        for layers in self.decoder.decoder:
            x = layers(x)
        return x, mu, log_var


class FeatureExtractorNet(nn.Module):
    def __init__(self, in_channel=1):
        super(FeatureExtractorNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(conv_layer("fe_l1", in_channel, 16))  # [1, 16, 16, 512]
        self.encoder.append(conv_layer("fe_l2", 16, 32, max_pooling=False))  # [1, 32, 16, 512]
        self.encoder.append(conv_layer("fe_l3", 32, 64))  # [1, 64, 8, 256]
        self.encoder.append(conv_layer("fe_l4", 64, 128, max_pooling=False))  # [1, 128, 8, 256]
        self.encoder.append(conv_layer("fe_l5", 128, 256))  # [1, 256, 4, 128]


class Decoder(nn.Module):
    def __init__(self, in_channel=1):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()
        self.decoder.append(de_conv_layer("de_fe_l1", 256, 128, decoder=True))  # [1, 16, 16, 512]
        self.decoder.append(de_conv_layer("de_fe_l2", 128, 64, max_pooled=False, decoder=True))  # [1, 32, 16, 512]
        self.decoder.append(de_conv_layer("de_fe_l3", 64, 32, decoder=True))  # [1, 64, 8, 256]
        self.decoder.append(de_conv_layer("de_fe_l4", 32, 16, max_pooled=False, decoder=True))  # [1, 128, 8, 256]
        self.decoder.append(de_conv_layer("de_fe_l5", 16, in_channel, decoder=True))  # [1, 256, 4, 128]


class CorrelationNet(nn.Module):
    def __init__(self, in_channel=256):
        super(CorrelationNet, self).__init__()
        self.correlation = nn.ModuleList()
        self._make_flow_prediction_layer(272, name="5_4")
        self._make_flow_prediction_layer(192, name="4_3", max_pooled=False)
        self._make_flow_prediction_layer(128, name="3_2")
        self._make_flow_prediction_layer(288, name="2_1", max_pooled=False)
        self._make_flow_prediction_layer(272, name="1_0")
        self.correlation.append(predict_flow_layer(1025))

    def _make_flow_prediction_layer(self, in_channel, name, max_pooled=True):
        pf_layer = predict_flow_layer(in_channel)
        upsample_pf_layer = de_conv_layer(f"flow_up_{name}", 2, 2, max_pooled=max_pooled)
        pf_seq = nn.Sequential(pf_layer, *upsample_pf_layer)
        self.correlation.append(pf_seq)


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
    out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


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
