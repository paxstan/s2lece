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

        # self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
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
        self.OS = params["OS"]
        self.model_type = params["type"]

        self.strides = [2, 2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

        # make the new stride
        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)

        else:
            # redo strides according to needed stride
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        model_blocks = {
            21: [1, 1, 2, 2, 1],
            53: [1, 2, 8, 8, 4],
        }

        self.blocks = model_blocks[params["model"]]

        self.conv1 = nn.Conv2d(self.in_channel, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(1, momentum=self.bn_d)
        if self.model_type == 'fe':
            self.norm1 = nn.InstanceNorm2d(1, momentum=self.bn_d)

        else:
            self.norm1 = nn.BatchNorm2d(1, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        # encoder
        self.enc1 = self._make_enc_layer(BasicBlock, [1, 2], self.blocks[0],
                                         stride=self.strides[0], model_type=self.model_type, bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(BasicBlock, [2, 4], self.blocks[1],
                                         stride=self.strides[1], model_type=self.model_type, bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(BasicBlock, [4, 8], self.blocks[2],
                                         stride=self.strides[2], model_type=self.model_type, bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(BasicBlock, [8, 16], self.blocks[3],
                                         stride=self.strides[3], model_type=self.model_type, bn_d=self.bn_d)
        # self.enc5 = self._make_enc_layer(BasicBlock, [16, 32], self.blocks[4],
        #                                  stride=self.strides[4], model_type=self.model_type, bn_d=self.bn_d)

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 32
        self.input_depth = 0

    @staticmethod
    def _make_enc_layer(block, planes, blocks, stride, model_type, bn_d=0.1):
        #  down sample
        layers = []
        layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                         kernel_size=3,
                                         stride=(1, stride), dilation=1,
                                         padding=1, bias=False)))
        if model_type == 'fe':
            layers.append(("instance", nn.InstanceNorm2d(planes[1], momentum=bn_d)))

        else:
            layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))

        layers.append(("relu", nn.LeakyReLU(0.1)))
        # layers = [,
        #           ("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)),
        #           ]

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
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]

        self.strides = [2, 1, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        # decoder
        self.dec5 = self._make_dec_layer(BasicBlock,
                                         [32, 16],
                                         bn_d=self.bn_d,
                                         stride=self.strides[0])
        self.dec4 = self._make_dec_layer(BasicBlock, [16, 16], bn_d=self.bn_d,
                                         stride=self.strides[1])
        self.dec3 = self._make_dec_layer(BasicBlock, [16, 8], bn_d=self.bn_d,
                                         stride=self.strides[2])
        self.dec2 = self._make_dec_layer(BasicBlock, [8, 4], bn_d=self.bn_d,
                                         stride=self.strides[3])
        self.dec1 = self._make_dec_layer(BasicBlock, [4, 2], bn_d=self.bn_d,
                                         stride=self.strides[4])

        # layer list to execute with skips
        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]
        # self.layers = [self.dec5, self.dec3, self.dec2, self.dec1]

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 2

    @staticmethod
    def _make_dec_layer(block, planes, bn_d=0.1, stride=2):
        layers = []

        #  down sample
        if stride == 2:
            layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                        kernel_size=(1, 4), stride=(1, 2),
                                                        padding=(0, 1))))
        else:
            layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                             kernel_size=3, padding=1)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
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

        # run layers
        x, skips, os = self.run_layer(x, self.dec5, skips, os)
        x, skips, os = self.run_layer(x, self.dec4, skips, os)
        x, skips, os = self.run_layer(x, self.dec3, skips, os)
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        x, skips, os = self.run_layer(x, self.dec1, skips, os)

        x = self.dropout(x)

        return x

    def get_last_depth(self):
        return self.last_channels


class AutoEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = FeatureExtractorNet(params)
        self.decoder = Decoder(params)
        self.head = nn.Sequential(nn.Dropout2d(p=0.01),
                                  nn.Conv2d(self.decoder.get_last_depth(),
                                            1, kernel_size=3,
                                            stride=1, padding=1))

    def forward(self, x):
        y, skips = self.encoder(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        return y


class ContextNet(nn.Module):
    def __init__(self, in_channel=1):
        super(ContextNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(
            conv_layer("fe_l1", in_channel, 2, max_pooling=False, instance_norm=True))  # [1, 16, 16, 512]
        self.encoder.append(conv_layer("fe_l2", 2, 4, instance_norm=True))  # [1, 32, 16, 512]
        self.encoder.append(conv_layer("fe_l3", 4, 8, instance_norm=True))  # [1, 64, 8, 256]
        self.encoder.append(conv_layer("fe_l4", 8, 16, instance_norm=True))  # [1, 128, 8, 256]
        self.encoder.append(conv_layer("fe_l5", 16, 32, instance_norm=True))  # [1, 256, 4, 128]

    def forward(self, x):
        i = 1
        net = {}
        inp = {}
        for layers in self.encoder:
            x = layers(x)
            x_net = torch.tanh(x)
            x_inp = torch.relu(x)
            net[i] = x_net
            inp[i] = x_inp
            i *= 2
        return net, inp
