import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.featurenet import FeatureExtractorNet, ContextNet, AutoEncoder
from models.attention import SelfAttention, CrossAttention
from models.update import BasicUpdateBlock
from models.utils import correlate, warp, linear_position_embedding_sine, de_conv_layer, Correlation1D, \
    PositionEmbeddingSine, coords_grid


class FlowModel(nn.Module):
    def __init__(self, config, device, encoder):
        super(FlowModel, self).__init__()
        self.config = config
        self.device = device
        self.to(self.device)
        self.encoder = encoder


class UpdateNet(nn.Module):
    def __init__(self, device, params):
        super().__init__()
        self.embedded_dim = params["embedded_dim"]
        self.down_sample_factor = params["downsample_factor"]
        self.iters = params["iters"]
        self.self_attn = True
        self.corr_attn = None
        self.flow_init = None
        self.self_attention = SelfAttention(embed_dim=self.embedded_dim, in_channel=4000)

        # self.self_attention = {
        #     16: SelfAttention(embed_dim=self.embedded_dim, in_channel=4000),
        #     4: SelfAttention(embed_dim=self.embedded_dim // 4, in_channel=16000),
        #     1: SelfAttention(embed_dim=self.embedded_dim // 4, in_channel=64000)
        #     # 16: nn.MultiheadAttention(embed_dim=self.embedded_dim, num_heads=10),
        #     # 4: nn.MultiheadAttention(embed_dim=self.embedded_dim // 4, num_heads=10),
        #     # 1: nn.MultiheadAttention(embed_dim=self.embedded_dim // 4, num_heads=10)
        #
        # }

        self.cross_attention = {
            16: CrossAttention(embed_dim=self.embedded_dim, in_channel_source=32, in_channel_target=32).to(device),
            4: CrossAttention(embed_dim=self.embedded_dim // 4, in_channel_source=32, in_channel_target=8).to(device),
            1: CrossAttention(embed_dim=self.embedded_dim // 8, in_channel_source=8, in_channel_target=2).to(device)

        }

        self.update_block = {
            16: BasicUpdateBlock(corr_channels=32,
                                 hidden_dim=self.embedded_dim,
                                 context_dim=32,
                                 downsample_factor=4).to(device),
            4: BasicUpdateBlock(corr_channels=8,
                                hidden_dim=self.embedded_dim // 4,
                                context_dim=self.embedded_dim // 4,
                                downsample_factor=4).to(device),
            1: BasicUpdateBlock(corr_channels=4,
                                hidden_dim=self.embedded_dim // 16,
                                context_dim=self.embedded_dim // 16,
                                downsample_factor=4, learn_upsample=False).to(device)

        }

        # self.self_attention1 = SelfAttention(input_dim=self.embedded_dim)
        # self.cross_attention1 = CrossAttention(input_dim=self.embedded_dim)
        #
        # self.self_attention2 = SelfAttention(input_dim=self.embedded_dim)
        # self.cross_attention2 = CrossAttention(input_dim=self.embedded_dim)
        #
        # self.self_attention3 = SelfAttention(input_dim=self.embedded_dim)
        # self.cross_attention3 = CrossAttention(input_dim=self.embedded_dim)

        # Update block
        # self.update_block = BasicUpdateBlock(corr_channels=self.embedded_dim,
        #                                      hidden_dim=self.embedded_dim,
        #                                      context_dim=self.embedded_dim,
        #                                      downsample_factor=4)

    def initialize_flow(self, img, downsample=None):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h, w // downsample_factor).to(img.device)
        coords1 = coords_grid(n, h, w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    @staticmethod
    def learned_upflow(flow, corr_attn, mask, downsample):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        n, _, h, w = flow.shape
        n2, c2, h2, w2 = corr_attn.shape
        mask = mask.view(n, 1, 9, 1, downsample, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(downsample * flow, kernel_size=3, padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_corr_attn = F.unfold(downsample * corr_attn, kernel_size=3, padding=1)
        up_corr_attn = up_corr_attn.view(n, c2, 9, 1, 1, h2, w2)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        up_corr_attn = torch.sum(mask * up_corr_attn, dim=2)
        up_corr_attn = up_corr_attn.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(n, 2, h, downsample * w), up_corr_attn.reshape(n, c2, h, downsample * w)

    def forward(self, feature1, feature2, net, inp, key, upsample=True):
        b, dim, h, w = feature1.size()
        coords0, coords1 = self.initialize_flow(feature1)
        if self.flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + self.flow_init

        coords1 = coords1.detach()  # stop gradient
        flow = coords1 - coords0

        if self.self_attn:
            corr_val = self.corr(feature1, feature2)
            self.corr_attn = self.self_attention(feature1.size(), corr_val)
            self.corr_attn = self.corr_attn.permute(0, 2, 1).view(feature1.size())
            self.self_attn = False
        else:
            self.corr_attn = self.cross_attention[key](self.corr_attn, feature1)
            b, h1, w1, = self.corr_attn.size()
            self.corr_attn = self.corr_attn.permute(0, 2, 1).view(b, w1, h, w)
        net, up_mask, delta_flow = self.update_block[key](net, inp, self.corr_attn, flow, upsample=True)
        coords1 = coords1 + delta_flow
        new_flow = coords1 - coords0

        if upsample:
            new_flow, self.corr_attn = self.learned_upflow(new_flow, self.corr_attn, up_mask, downsample=4)
        self.flow_init = new_flow
        return self.flow_init


class SleceNet(nn.Module):
    def __init__(self, config, device, params, encoder, train=True):
        super().__init__(config, device, train)
        self.config = config
        self.device = device
        self.to(self.device)
        self.encoder = encoder
        self.context_net = ContextNet().float()
        self.update = UpdateNet(device, params)

    def calculate_n_parameters(self):
        def times(shape):
            parameters = 1
            for layer in list(shape):
                parameters *= layer
            return parameters

        layer_params = [times(x.size()) for x in list(self.parameters())]

        return sum(layer_params)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr_volume = torch.matmul(fmap1.transpose(1, 2), fmap2)
        # corr = corr.view(batch, ht, wd, 1, ht, wd)
        # corr_volume = corr_volume.view(batch, ht, wd, ht, wd)
        # return corr_volume / torch.sqrt(torch.tensor(dim).float())
        min_val = torch.min(corr_volume)
        max_val = torch.max(corr_volume)
        corr_volume = (corr_volume - min_val) / (max_val - min_val)  # normalize the correlation
        corr_volume = 2 * corr_volume - 1  # set in the range of -1 and 1

        return corr_volume

    def extract_feature(self, x1, x2):
        # x1_fe_out = [x1]
        # x2_fe_out = [x2]

        x1, x1_skips = self.encoder(x1)
        x2, x2_skips = self.encoder(x2)

        # for layers in self.feature_net.encoder:
        #     x1 = layers(x1)
        #     x2 = layers(x2)
        #     x1_fe_out.append(x1)
        #     x2_fe_out.append(x2)

        return x1, x2, x1_skips, x2_skips

    # def extract_context(self, x):
    #     for layers in self.context_net.encoder:
    #         x = layers(x)
    #     return x

    def forward(self, x1, x2, flow_init=None):
        # print(x1.shape)
        self_attn = True
        corr_attn = None
        upsample = True
        net, inp = self.context_net(x1)  # extract context from x1
        # net = torch.tanh(x1_context)
        # inp = torch.relu(x1_context)

        # f1, f2, x1_skips, x2_skips = self.extract_feature(x1, x2)  # extract features from x1 and x2
        f1, x1_skips = self.encoder(x1)
        f2, x2_skips = self.encoder(x2)
        x1_skips[16] = f1
        x2_skips[16] = f2

        keys = list(x1_skips.keys())
        keys.reverse()

        keys = [16, 4, 1]

        for key in keys:
            feature1 = x1_skips[key]
            feature2 = x2_skips[key]
            if key != keys[-1]:
                upsample = False

            flow = self.update(feature1, feature2, net[key], inp [key], key, upsample=upsample)

            # b, dim, h, w = feature1.size()
            #
            # coords0, coords1 = self.initialize_flow(x1, downsample=key)
            # if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            #     coords1 = coords1 + flow_init
            #
            # coords1 = coords1.detach()  # stop gradient
            # flow = coords1 - coords0
            #
            # if self_attn:
            #     corr_val = self.corr(feature1, feature2)
            #     corr_attn = self.self_attention(feature1.size(), corr_val)
            #     corr_attn = corr_attn.permute(0, 2, 1).view(feature1.size())
            #     self_attn = False
            # else:
            #     corr_attn = self.cross_attention[key](corr_attn, feature1)
            #     b, h1, w1, = corr_attn.size()
            #     corr_attn = corr_attn.permute(0, 2, 1).view(b, w1, h, w)
            #
            # net[key], up_mask, delta_flow = self.update_block[key](net[key], inp[key], corr_attn, flow, upsample=True)
            # coords1 = coords1 + delta_flow
            # new_flow = coords1 - coords0
            #
            # if key != keys[-1]:
            #     new_flow, corr_attn = self.learned_upflow(new_flow, corr_attn, up_mask, downsample=4)
            # flow_init = new_flow
            # print(f"flow_init : {flow_init.device.type}")
            # print(f"corr : {corr_attn.device.type}")
            # # corr_attn = corr_attn.to(

        # for i in range(self.iters):
        #     coords1 = coords1.detach()  # stop gradient
        #     flow = coords1 - coords0
        #
        #     feature2_attn = self.cross_attention_layer(feature1, feature2)
        #     b, h1, w1, = feature2_attn.size()
        #     feature2_attn = feature2_attn.permute(0, 2, 1).view(b, w1, h, w)
        #
        #     corr_val = self.corr(feature1, feature2_attn)
        #     b, h1, w1, = corr_val.size()
        #
        #     corr_attention = self.self_attention_layer(corr_val)
        #     corr_attention = corr_attention.permute(0, 2, 1).view(b, self.embedded_dim, h, w)
        #
        #     net, up_mask, delta_flow = self.update_block(net, inp, corr_attention, flow, upsample=True)
        #     coords1 = coords1 + delta_flow
        #
        #     flow_up = self.learned_upflow(coords1 - coords0, up_mask)
        #     # x_img = flow_up[:, :, 0].reshape(-1).astype(int)
        #     # y_img = flow_up[:, :, 1].reshape(-1).astype(int)
        #     # mask_valid_in_2 = torch.zeros_like(flow_up)
        #     # mask_in_bound = (y_img >= 0) * (y_img < h) * (x_img >= 0) * (x_img < w)
        #     # mask_valid_in_2[mask_in_bound] = mask2[y_img[mask_in_bound], x_img[mask_in_bound]]
        #     flow_predictions.append(flow_up)

        return flow_init
