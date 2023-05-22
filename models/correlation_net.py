import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spatial_correlation_sampler import spatial_correlation_sample
from visualization.visualization import flow2rgb, display_flows


class CorrelationNetwork(nn.Module):
    def __init__(self, patch_size=1):
        super(CorrelationNetwork, self).__init__()
        self.conv_redir = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 256. 32, 1024]

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), stride=(2, 4), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 512, 16, 256]

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 512, 16, 256]

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(1, 3), stride=(2, 4), padding=(0, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 1024, 8, 64]

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 1024, 8, 64]

        self.predict_flow5 = nn.Conv2d(1024, 2, kernel_size=(1, 3),
                                       stride=(1, 1), padding=(0, 1))  # [1, 2, 8, 64]

        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(in_channels=2, out_channels=2,
                                                       kernel_size=(1, 3), stride=(2, 4),
                                                       padding=(0, 1), output_padding=(1, 3))  # [1, 2, 15, 14]
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(1, 3), stride=(2, 4), padding=(0, 1), output_padding=(1, 3)),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 512, 15, 13]

        self.predict_flow4 = nn.Conv2d(1026, 2, kernel_size=(1, 3),
                                       stride=(1, 1), padding=(0, 1))

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=2, out_channels=2,
                                                       kernel_size=(1, 3), stride=(2, 4),
                                                       padding=(0, 1), output_padding=(1, 3))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(1, 3), stride=(2, 4), padding=(0, 1), output_padding=(1, 3)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.predict_flow3 = nn.Conv2d(514, 2, kernel_size=(1, 3),
                                       stride=(1, 1), padding=(0, 1))

        self.predict_flow_truth = nn.Conv2d(3, 2, kernel_size=(1, 1),
                                            stride=(1, 1), padding=(0, 1))

    @staticmethod
    def predict_flow(in_planes):
        return nn.Conv2d(in_planes, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)

    @staticmethod
    def crop_like(input, target):
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :, :target.size(2), :target.size(3)]

    @staticmethod
    def visualize_flow(flow):
        rgb_flow = flow2rgb(20 * flow, max_value=None)
        img = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
        return img

    @staticmethod
    def correlate(input1, input2, patch_size=32, stride=32):
        correlation_tensor = torch.Tensor()
        _, c, h, w = input1.size()
        for i in range(0, w - patch_size + 1, stride):
            patch_input2 = input2[:, :, :, i: i + patch_size]
            patch_input2 = patch_input2.repeat(1, 1, 1, 32)
            # each pixel in input one against all pixel in input2 -> produce a channel
            out_corr2 = spatial_correlation_sample(input1,
                                                   patch_input2,
                                                   kernel_size=1,
                                                   patch_size=1,
                                                   stride=1,
                                                   padding=0,
                                                   dilation_patch=2)
            # collate dimensions 1 and 2 in order to be treated as a
            # regular 4D tensor
            b, ph, pw, h, w = out_corr2.size()
            out_corr2 = out_corr2.view(b, ph * pw, h, w) / input1.size(1)  # [1, 1, 32, 1024]
            # out_corr = F.conv2d(input1, patch_input2, stride=32, padding=0)
            F.leaky_relu_(out_corr2, 0.1)
            correlation_tensor = torch.concat((correlation_tensor, out_corr2), dim=1)
        return correlation_tensor  # [1, 32, 32, 1024]

    def forward(self, x1, x2, og_flow, flow):
        flow_gt = self.predict_flow_truth(flow)
        out_correlation = self.correlate(x1, x2)
        x1_redir = self.conv_redir(x1)

        out_concat = torch.cat([x1_redir, out_correlation], dim=1)  # [1, 64, 32, 1024]

        out_conv3 = self.conv3_1(out_concat)  # [1, 256, 32, 1024]
        out_conv4 = self.conv4_1(self.conv4(out_conv3))  # [1, 512, 16, 256]
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # [1, 1024, 8, 64]

        flow5 = self.predict_flow5(out_conv5)  # [1, 2, 8, 64]
        flow5_up = self.crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)  # [1, 2, 16, 256]
        out_deconv4 = self.crop_like(self.deconv5(out_conv5), out_conv4)  # [1, 512, 16, 256]

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)  # [1, 1026, 16, 256]

        flow4 = self.predict_flow4(concat4)  # [1, 2, 32, 1024]
        flow4_up = self.crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)  # [1, 256, 32, 1024]
        out_deconv3 = self.crop_like(self.deconv4(out_conv4), out_conv3)  # [1, 256, 32, 1024]

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)  # [1, 514, 32, 1024]
        pred_flow = self.predict_flow3(concat3)  # [1, 2, 32, 1024]

        display_flows(original_flow=og_flow, flow_projected_2=flow_gt, predicted_flow=pred_flow)

        return pred_flow
