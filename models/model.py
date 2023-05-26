import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spatial_correlation_sampler import spatial_correlation_sample
from visualization.visualization import flow_to_color


class FlowModel(nn.Module):
    def __init__(self, device):
        super(FlowModel, self).__init__()
        self.device = device
        self.to(self.device)
        self.feature_net = FeatureExtractorNet().to(self.device)
        self.correlation_net = CorrelationNet().to(self.device)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, x1, x2):
        x1_fe_out1 = self.feature_net.layer1(x1)
        x1_fe_out2 = self.feature_net.layer2(x1_fe_out1)
        x1_fe_out3 = self.feature_net.layer3(x1_fe_out2)
        x1_fe_out4 = self.feature_net.layer4(x1_fe_out3)
        x1_fe_out5 = self.feature_net.layer5(x1_fe_out4)

        x2_fe_out1 = self.feature_net.layer1(x2)
        x2_fe_out2 = self.feature_net.layer2(x2_fe_out1)
        x2_fe_out3 = self.feature_net.layer3(x2_fe_out2)
        x2_fe_out4 = self.feature_net.layer4(x2_fe_out3)
        x2_fe_out5 = self.feature_net.layer5(x2_fe_out4)

        x1_x2_correlate_5 = correlate(x1_fe_out5, x2_fe_out5, patch_size=4, dilation_patch=2)  # [1, 16, 4, 128]
        x1_c5_concat = torch.cat([x1_fe_out5, x1_x2_correlate_5], dim=1)  # [1, 272, 4, 128]
        predict_flow5 = self.correlation_net.predict_flow5(x1_c5_concat)  # [1, 2, 4, 128]
        # flow_img_5 = flow_to_color(predict_flow5.detach().squeeze().numpy().transpose(1, 2, 0))
        predict_flow5_up = self.correlation_net.up_sample_flow5_4(predict_flow5)
        wrapped_x2_out4 = warp(x2_fe_out4, predict_flow5_up)

        x1_x2_correlate_4 = correlate(x1_fe_out4, wrapped_x2_out4, patch_size=8, dilation_patch=4)  # [1, 64, 8, 256]
        x1_c4_concat = torch.cat([x1_fe_out4, x1_x2_correlate_4], dim=1)  # [1, 192, 8, 256]
        predict_flow4 = self.correlation_net.predict_flow4(x1_c4_concat)  # [1, 2, 8, 256]
        # flow_img_4 = flow_to_color(predict_flow4.detach().squeeze().numpy().transpose(1, 2, 0))
        predict_flow4_up = self.correlation_net.up_sample_flow4_3(predict_flow4)
        wrapped_x2_out3 = warp(x2_fe_out3, predict_flow4_up)

        x1_x2_correlate_3 = correlate(x1_fe_out3, wrapped_x2_out3, patch_size=8, dilation_patch=4)  # [1, 64, 8, 256]
        x1_c3_concat = torch.cat([x1_fe_out3, x1_x2_correlate_3], dim=1)  # [1, 128, 8, 256]
        predict_flow3 = self.correlation_net.predict_flow3(x1_c3_concat)  # [1, 2, 8, 256]
        # flow_img_3 = flow_to_color(predict_flow3.detach().squeeze().numpy().transpose(1, 2, 0))
        predict_flow3_up = self.correlation_net.up_sample_flow3_2(predict_flow3)
        wrapped_x2_out2 = warp(x2_fe_out2, predict_flow3_up)

        x1_x2_correlate_2 = correlate(x1_fe_out2, wrapped_x2_out2, patch_size=16, dilation_patch=8)  # [1, 256, 16, 512]
        x1_c2_concat = torch.cat([x1_fe_out2, x1_x2_correlate_2], dim=1)  # [1, 286, 16, 512]
        predict_flow2 = self.correlation_net.predict_flow2(x1_c2_concat)  # [1, 2, 16, 512]
        # flow_img_2 = flow_to_color(predict_flow2.detach().squeeze().numpy().transpose(1, 2, 0))
        predict_flow2_up = self.correlation_net.up_sample_flow2_1(predict_flow2)
        wrapped_x2_out1 = warp(x2_fe_out1, predict_flow2_up)

        x1_x2_correlate_1 = correlate(x1_fe_out1, wrapped_x2_out1, patch_size=16, dilation_patch=8)  # [1, 256, 16, 512]
        x1_c1_concat = torch.cat([x1_fe_out1, x1_x2_correlate_1], dim=1)  # [1, 286, 16, 512]
        predict_flow1 = self.correlation_net.predict_flow1(x1_c1_concat)  # [1, 2, 16, 512]
        # flow_img_1 = flow_to_color(predict_flow1.detach().squeeze().numpy().transpose(1, 2, 0))
        predict_flow1_up = self.correlation_net.up_sample_flow1_0(predict_flow1)
        wrapped_x2 = warp(x2, predict_flow1_up)

        x1_x2_correlate_0 = correlate(x1, wrapped_x2, patch_size=32, dilation_patch=16)  # [1, 256, 16, 512]
        x1_c0_concat = torch.cat([x1, x1_x2_correlate_0], dim=1)  # [1, 286, 16, 512]
        predict_flow0 = self.correlation_net.predict_flow0(x1_c0_concat)  # [1, 2, 16, 512]
        # flow_img_ = flow_to_color(predict_flow0.detach().squeeze().numpy().transpose(1, 2, 0))

        return predict_flow0


class FeatureExtractorNet(nn.Module):
    def __init__(self, in_channel=1):
        super(FeatureExtractorNet, self).__init__()
        self.count = 1
        self.layer1 = conv_layer("fe_l1", in_channel, 16)  # [1, 16, 16, 512]
        self.layer2 = conv_layer("fe_l2", 16, 32, max_pooling=False)  # [1, 32, 16, 512]
        self.layer3 = conv_layer("fe_l3", 32, 64)  # [1, 64, 8, 256]
        self.layer4 = conv_layer("fe_l4", 64, 128, max_pooling=False)  # [1, 128, 8, 256]
        self.layer5 = conv_layer("fe_l5", 128, 256)  # [1, 256, 4, 128]

        # self.de_layer5 = de_conv_layer("de_fe_l1", 256, 128)  # [1, 16, 16, 512]
        # self.de_layer4 = de_conv_layer("de_fe_l2", 128, 64, max_pooled=False)  # [1, 32, 16, 512]
        # self.de_layer3 = de_conv_layer("de_fe_l3", 64, 32)  # [1, 64, 8, 256]
        # self.de_layer2 = de_conv_layer("de_fe_l4", 32, 16, max_pooled=False)  # [1, 128, 8, 256]
        # self.de_layer1 = de_conv_layer("de_fe_l5", 16, in_channel)  # [1, 256, 4, 128]


class CorrelationNet(nn.Module):
    def __init__(self, in_channel=256):
        super(CorrelationNet, self).__init__()
        self.correlate_layer = conv_layer("cn_re", in_channel, 32)
        self.conv3 = conv_layer("cn_c3", 1056, 256)
        self.conv4 = conv_layer("cn_c4", 256, 512)
        self.conv4_1 = conv_layer("cn_c4_1", 512, 512)
        self.conv5 = conv_layer("cn_c5", 512, 1024)
        self.conv5_1 = conv_layer("cn_c5_1", 1024, 1024)

        self.predict_flow5 = predict_flow(272)
        self.predict_flow4 = predict_flow(192)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(288)
        self.predict_flow1 = predict_flow(272)
        self.predict_flow0 = predict_flow(1025)

        self.up_sample_flow5_4 = de_conv_layer("flow_up5_4", 2, 2)
        self.up_sample_flow4_3 = de_conv_layer("flow_up4_3", 2, 2, max_pooled=False)
        self.up_sample_flow3_2 = de_conv_layer("flow_up3_2", 2, 2)
        self.up_sample_flow2_1 = de_conv_layer("flow_up2_1", 2, 2, max_pooled=False)
        self.up_sample_flow1_0 = de_conv_layer("flow_up1_0", 2, 2)


def conv_layer(name, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), max_pooling=True):
    seq_model = nn.Sequential()
    seq_model.add_module(f"conv_{name}",
                         nn.Conv2d(in_channel, out_channel,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2), bias=False))
    seq_model.add_module(f"bn_{name}", nn.BatchNorm2d(out_channel))
    seq_model.add_module(f"activ_{name}", nn.LeakyReLU(0.1, inplace=True))
    if max_pooling:
        seq_model.add_module(f"max_pool_{name}", nn.MaxPool2d(kernel_size=2, stride=2))
    return seq_model


def de_conv_layer(name, in_channel, out_channel, kernel_size=(3, 3), max_pooled=True):
    seq_model = nn.Sequential()
    if max_pooled:
        seq_model.add_module(f"de_conv_{name}",
                             nn.ConvTranspose2d(in_channel, out_channel,
                                                kernel_size=kernel_size, stride=2,
                                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                                output_padding=(1, 1),
                                                bias=False))
    else:
        seq_model.add_module(f"de_conv_{name}",
                             nn.ConvTranspose2d(in_channel, out_channel,
                                                kernel_size=kernel_size, stride=1,
                                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                                bias=False))
    return seq_model


def predict_flow(in_planes):
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
    mask = torch.autograd.Variable(torch.ones(x.size()))

    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask
