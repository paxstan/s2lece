import torch.nn as nn
import torch
from collections import OrderedDict
from models.model_utils import conv_layer, de_conv_layer
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, model_type, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        if model_type == 'fe':
            self.norm1 = nn.InstanceNorm2d(planes[0], momentum=bn_d)

        else:
            self.norm1 = nn.BatchNorm2d(planes[0], momentum=bn_d)

        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)

        if model_type == 'fe':
            self.norm2 = nn.InstanceNorm2d(planes[1], momentum=bn_d)

        else:
            self.norm2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out += residual
        return out


class FeatureExtractorNet(nn.Module):
    """
         Class for DarknetSeg. Subclasses PyTorch's own "nn" module
      """

    def __init__(self, params):
        super(FeatureExtractorNet, self).__init__()
        self.in_channel = params["in_channel"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.model_type = params["type"]

        self.conv1 = nn.Conv2d(self.in_channel, 16, kernel_size=3,
                               stride=(1, 1), padding=1, bias=False)

        if self.model_type == 'fe':
            self.norm1 = nn.InstanceNorm2d(16, momentum=self.bn_d)

        else:
            self.norm1 = nn.BatchNorm2d(16, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        # encoder
        self.enc1 = self._make_enc_layer(BasicBlock, [16, 32], 1,
                                         stride=(1, 2), model_type=self.model_type, bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(BasicBlock, [32, 64], 2,
                                         stride=(1, 2), model_type=self.model_type, bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(BasicBlock, [64, 128], 2,
                                         stride=(1, 2), model_type=self.model_type, bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(BasicBlock, [128, 256], 1,
                                         stride=(1, 2), model_type=self.model_type, bn_d=self.bn_d)
        # self.enc5 = self._make_enc_layer(BasicBlock, [256, 512], 1,
        #                                  stride=(1, 2), model_type=self.model_type, bn_d=self.bn_d)

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 1024
        self.input_depth = 0

    @staticmethod
    def _make_enc_layer(block, planes, blocks, stride, model_type, bn_d=0.1):
        #  down sample
        layers = []
        layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                         kernel_size=3,
                                         stride=stride, dilation=1,
                                         padding=1, bias=False)))
        if model_type == 'fe':
            layers.append(("instance", nn.InstanceNorm2d(planes[1], momentum=bn_d)))

        else:
            layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))

        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        in_planes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), block(in_planes, planes, model_type, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    @staticmethod
    def run_layer(x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth

    def forward(self, x):
        # store for skip connections
        skips = {}
        os = 1

        # first layer
        x, skips, os = self.run_layer(x, self.conv1, skips, os)
        x, skips, os = self.run_layer(x, self.norm1, skips, os)
        x, skips, os = self.run_layer(x, self.relu1, skips, os)

        # all encoder blocks with intermediate dropouts
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        # x, skips, os = self.run_layer(x, self.enc5, skips, os)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)

        return x, skips


class Decoder(nn.Module):
    def __init__(self, params, OS=16, feature_depth=125):
        super(Decoder, self).__init__()
        self.backbone_OS = 16
        self.backbone_feature_depth = feature_depth
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]

        # decoder
        # self.dec5 = self._make_dec_layer(BasicBlock, [512, 256], bn_d=self.bn_d,
        #                                  stride=(1, 2))
        self.dec4 = self._make_dec_layer(BasicBlock, [256, 128], bn_d=self.bn_d,
                                         stride=(1, 2))
        self.dec3 = self._make_dec_layer(BasicBlock, [128, 64], bn_d=self.bn_d,
                                         stride=(1, 2))
        self.dec2 = self._make_dec_layer(BasicBlock, [64, 32], bn_d=self.bn_d,
                                         stride=(1, 2))
        self.dec1 = self._make_dec_layer(BasicBlock, [32, 16], bn_d=self.bn_d,
                                         stride=(1, 2))
        self.de_conv = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=(1, 1),
                                          padding=(1, 1))

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 1

    @staticmethod
    def _make_dec_layer(block, planes, stride, bn_d=0.1):
        layers = []

        layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                    kernel_size=(1, 4), stride=stride,
                                                    padding=(0, 1))))
        layers.append(("bn", nn.InstanceNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        if block is not None:
            layers.append(("residual", block(planes[1], planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    @staticmethod
    def run_layer(x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os].detach()  # add skip
        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = self.backbone_OS
        # x, skips, os = self.run_layer(x, self.dec5, skips, os)
        x, skips, os = self.run_layer(x, self.dec4, skips, os)
        x, skips, os = self.run_layer(x, self.dec3, skips, os)
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        x, skips, os = self.run_layer(x, self.dec1, skips, os)
        x = self.de_conv(x)

        x = self.dropout(x)

        return x

    def get_last_depth(self):
        return self.last_channels


class AutoEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = FeatureExtractorNet(params)
        self.decoder = Decoder(params)
        # self.head = nn.Sequential(nn.Dropout2d(p=0.01),
        #                           nn.Conv2d(self.decoder.get_last_depth(),
        #                                     1, kernel_size=3,
        #                                     stride=(1, 2), padding=1))

    def calculate_n_parameters(self):
        def times(shape):
            parameters = 1
            for layer in list(shape):
                parameters *= layer
            return parameters

        layer_params = [times(x.size()) for x in list(self.parameters())]

        return sum(layer_params)

    def forward(self, x):
        y, skips = self.encoder(x)
        y = self.decoder(y, skips)
        # y = self.head(y)
        return y
