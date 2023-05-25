import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import numpy as np
from utils.common import patch_extractor, compare_images


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

    @staticmethod
    def view_result(output_tensor):
        # Convert the output tensor to a numpy array for visualization
        output_array = output_tensor.detach().numpy()
        # Scale the pixel values of the output array to the range [0, 1]
        output_array = (output_array - output_array.min()) / (output_array.max() - output_array.min())
        plt.imshow(output_array.squeeze())
        plt.show()

    @staticmethod
    def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )

    @staticmethod
    def de_conv(batch_norm, in_planes, out_planes):
        if batch_norm:
            return nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )

    def forward(self, x):
        x_patch = patch_extractor(x)
        out_encoded = torch.Tensor()
        for patch in x_patch:
            out_encoded = torch.concat((out_encoded, self.encoder(patch)), dim=-1)
        # out_encoder = [self.encoder(patch) for patch in x_patch]
        # compare_images(x_patch[0], out_encoder[0])
        # self.view_result(x.clone().permute(1, 3, 2, 0))
        # out_decoder = [self.decoder(patch) for patch in out_encoder]
        # compare_images(x_patch[0], out_decoder[0])
        # self.view_result(x.clone())
        return out_encoded


class ConvolutionAE(Autoencoder):
    def __init__(self):
        Autoencoder.__init__(self)
        self.encoder = nn.Sequential(
            # for 32*32
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # for 32*1024
            # # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # # nn.BatchNorm2d(128),
            # # nn.LeakyReLU(0.1, inplace=True),
            #
            # # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.1, inplace=True),
            #
            # # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.1, inplace=True),
            #
            # # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.1, inplace=True),
            #
            # nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Tanh()
        )


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer1 = self.conv_layer(1, 16)
        self.layer2 = self.conv_layer(16, 32)
        self.layer3 = self.conv_layer(32, 64)
        self.layer4 = self.conv_layer(64, 128)
        self.layer5 = self.conv_layer(128, 256)

    @staticmethod
    def conv_layer(in_channel, out_channel, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x, device):
        x_patch = patch_extractor(x)
        out_encoded = torch.Tensor().to(device)
        for patch in x_patch:
            patch = self.layer1(patch)
            patch = self.layer2(patch)
            patch = self.layer3(patch)
            patch = self.layer4(patch)
            patch = self.layer5(patch)
            out_encoded = torch.concat((out_encoded, patch), dim=-1)
        return out_encoded
