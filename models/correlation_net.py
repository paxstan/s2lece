import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import spatial_correlation_sample
from visualization.visualization import flow_to_color


class CorrelationNetwork(nn.Module):
    def __init__(self, patch_size=1):
        super(CorrelationNetwork, self).__init__()
        self.corr_redir = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(473, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 256. 30, 63]

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), stride=(2, 4), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 512, 15, 16]

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 512, 15, 16]

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(1, 3), stride=(2, 4), padding=(0, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 1024, 8, 4]

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 1024, 8, 4]

        self.predict_flow5 = self.predict_flow(1024)  # [1, 2, 8, 4]
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(in_channels=2, out_channels=2,
                                                       kernel_size=(1, 3), stride=(2, 4),
                                                       padding=(0, 1), output_padding=(0, 1))  # [1, 2, 15, 14]
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(1, 3), stride=(2, 4), padding=(0, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )  # [1, 512, 15, 13]

        self.predict_flow4 = self.predict_flow(770)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.predict_flow3 = self.predict_flow(386)

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
        flow = flow.detach().squeeze().numpy().transpose(1, 2, 0)
        img = flow_to_color(flow)
        plt.imshow(img)

    @staticmethod
    def correlate(input1, input2):
        out_corr = spatial_correlation_sample(input1,
                                              input2,
                                              kernel_size=1,
                                              patch_size=21,
                                              stride=1,
                                              padding=0,
                                              dilation_patch=2)
        # collate dimensions 1 and 2 in order to be treated as a
        # regular 4D tensor
        b, ph, pw, h, w = out_corr.size()
        out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
        return F.leaky_relu_(out_corr, 0.1)

    def forward(self, x1, x2):
        # out = F.conv2d(x1, x2)
        # mean1 = x1.mean(dim=(0, 1))
        # mean2 = x2.mean(dim=(0, 1))
        #
        # std1 = x1.std(dim=(0, 1))
        # std2 = x2.std(dim=(0, 1))
        #
        # normal_x1 = x1 - mean1
        # normal_x2 = x2 - mean2
        #
        # out = F.conv2d(input=normal_x1, weight=normal_x2[:, :, :, :20])

        x1_redir = self.corr_redir(x1)
        out_correlation = self.correlate(x1, x2)
        out_concat = torch.cat([x1_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(out_concat)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # out_conv4_1 = self.conv4_1(out_conv4)
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        flow5 = self.predict_flow5(out_conv5)
        flow5_up = self.crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        self.visualize_flow(flow5_up)
        out_deconv4 = self.crop_like(self.deconv5(out_conv5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)  #
        self.visualize_flow(flow4_up)
        out_deconv3 = self.crop_like(self.deconv4(out_conv4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        self.visualize_flow(flow3)

        return out_correlation
