import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.featurenet import FeatureExtractorNet, ContextNet, AutoEncoder
from models.attention import SelfAttention, CrossAttention
from models.update import BasicUpdateBlock
from models.model_utils import coords_grid


class FlowModel(nn.Module):
    def __init__(self, config, device, encoder):
        super(FlowModel, self).__init__()
        self.config = config
        self.device = device
        self.to(self.device)
        self.encoder = encoder


class UpdateNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        embedded_dim = params["embedded_dim"]
        # self.down_sample_factor = params["downsample_factor"]
        # self.iters = params["iters"]
        # self.self_attn = True
        # self.corr_attn = None
        # self.flow_init = None
        self.self_attention = SelfAttention(embed_dim=embedded_dim, in_channel=4000)

        # self.cross_attention_16 = CrossAttention(embed_dim=embedded_dim,
        #                                          in_channel_source=32, in_channel_target=32)
        self.update_block_16 = BasicUpdateBlock(corr_channels=32,
                                                hidden_dim=embedded_dim,
                                                context_dim=32,
                                                downsample_factor=4)

        self.cross_attention_4 = CrossAttention(embed_dim=embedded_dim // 4,
                                                in_channel_source=32, in_channel_target=8)

        self.update_block_4 = BasicUpdateBlock(corr_channels=8,
                                               hidden_dim=embedded_dim // 4,
                                               context_dim=8,
                                               downsample_factor=4)

        self.cross_attention_1 = CrossAttention(embed_dim=embedded_dim // 8,
                                                in_channel_source=8, in_channel_target=2)

        self.update_block_1 = BasicUpdateBlock(corr_channels=4,
                                               hidden_dim=embedded_dim // 16,
                                               context_dim=2,
                                               downsample_factor=4, learn_upsample=False)

        # self.cross_attention = { 16: CrossAttention(embed_dim=self.embedded_dim, in_channel_source=32,
        # in_channel_target=32),  # .to(device), 4: CrossAttention(embed_dim=self.embedded_dim // 4,
        # in_channel_source=32, in_channel_target=8), # .to(device), 1: CrossAttention(embed_dim=self.embedded_dim //
        # 8, in_channel_source=8, in_channel_target=2)  # .to(device)
        #
        # }
        #
        # self.update_block = {
        #     16: BasicUpdateBlock(corr_channels=32,
        #                          hidden_dim=self.embedded_dim,
        #                          context_dim=32,
        #                          downsample_factor=4),  # .to(device),
        #     4: BasicUpdateBlock(corr_channels=8,
        #                         hidden_dim=self.embedded_dim // 4,
        #                         context_dim=self.embedded_dim // 4,
        #                         downsample_factor=4),  # .to(device),
        #     1: BasicUpdateBlock(corr_channels=4,
        #                         hidden_dim=self.embedded_dim // 16,
        #                         context_dim=self.embedded_dim // 16,
        #                         downsample_factor=4, learn_upsample=False)  # .to(device)
        #               }

    @staticmethod
    def initialize_flow(img, downsample=1):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        coords0 = coords_grid(n, h, w // downsample).to(img.device)
        coords1 = coords_grid(n, h, w // downsample).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr_volume = torch.matmul(fmap1.transpose(1, 2), fmap2)
        min_val = torch.min(corr_volume)
        max_val = torch.max(corr_volume)
        corr_volume = (corr_volume - min_val) / (max_val - min_val)  # normalize the correlation
        corr_volume = 2 * corr_volume - 1  # set in the range of -1 and 1

        return corr_volume

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

    # def forward(self, flow_init, feature1, feature2, net, inp, key, upsample=True):
    #     b, dim, h, w = feature1.size()
    #     coords0, coords1 = self.initialize_flow(feature1)
    #     if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
    #         coords1 = coords1 + flow_init
    #
    #     coords1 = coords1.detach()  # stop gradient
    #     flow = coords1 - coords0
    #
    #     if self.self_attn:
    #         corr_val = self.corr(feature1, feature2)
    #         self.corr_attn = self.self_attention(feature1.size(), corr_val)
    #         self.corr_attn = self.corr_attn.permute(0, 2, 1).view(feature1.size())
    #         self.self_attn = False
    #     else:
    #         self.corr_attn = self.cross_attention[key](self.corr_attn, feature1)
    #         b, h1, w1, = self.corr_attn.size()
    #         self.corr_attn = self.corr_attn.permute(0, 2, 1).view(b, w1, h, w)
    #     net, up_mask, delta_flow = self.update_block[key](net, inp, self.corr_attn, flow, upsample=True)
    #     coords1 = coords1 + delta_flow
    #     new_flow = coords1 - coords0
    #
    #     if upsample:
    #         new_flow, self.corr_attn = self.learned_upflow(new_flow, self.corr_attn, up_mask, downsample=4)
    #
    #     return new_flow

    def forward(self, feature1, feature2, net1, inp1, flow_init=None):

        # key 16
        coords0, coords1 = self.initialize_flow(feature1[0])
        if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + flow_init

        coords0 = coords0.detach()
        coords1 = coords1.detach()  # stop gradient
        flow = coords1 - coords0

        corr_val = self.corr(feature1[0], feature2[0])
        corr_attn = self.self_attention(feature1[0].size(), corr_val)
        corr_attn = corr_attn.permute(0, 2, 1).view(feature1[0].size())

        net, up_mask, delta_flow = self.update_block_16(net1[0], inp1[0], corr_attn, flow, upsample=True)
        coords1 = coords1 + delta_flow
        flow = coords1 - coords0
        flow, corr_attn = self.learned_upflow(flow, corr_attn, up_mask, downsample=4)

        # key 4

        _, _, h, w = feature1[1].size()
        coords0, coords1 = self.initialize_flow(feature1[1])

        coords1 = coords1 + flow

        coords0 = coords0.detach()
        coords1 = coords1.detach()  # stop gradient
        flow = coords1 - coords0

        corr_attn = self.cross_attention_4(corr_attn, feature1[1])
        b, h1, w1, = corr_attn.size()
        corr_attn = corr_attn.permute(0, 2, 1).view(b, w1, h, w)

        net, up_mask, delta_flow = self.update_block_4(net1[1], inp1[1], corr_attn, flow, upsample=True)
        coords1 = coords1 + delta_flow
        flow = coords1 - coords0
        flow, corr_attn = self.learned_upflow(flow, corr_attn, up_mask, downsample=4)

        # key 1

        _, _, h, w = feature1[2].size()
        coords0, coords1 = self.initialize_flow(feature1[2])

        coords1 = coords1 + flow

        coords0 = coords0.detach()
        coords1 = coords1.detach()  # stop gradient
        flow = coords1 - coords0

        corr_attn = self.cross_attention_1(corr_attn, feature1[2])
        b, h1, w1, = corr_attn.size()
        corr_attn = corr_attn.permute(0, 2, 1).view(b, w1, h, w)

        net, up_mask, delta_flow = self.update_block_1(net1[2], inp1[2], corr_attn, flow, upsample=True)
        coords1 = coords1 + delta_flow
        flow = coords1 - coords0

        return flow


class S2leceNet(nn.Module):
    def __init__(self, config, fe_params, update_params):
        super().__init__()
        self.config = config
        self.keys = [16, 4, 1]
        self.context_net = ContextNet().float()
        self.encoder = FeatureExtractorNet(fe_params)
        self.update = UpdateNet(update_params)

    def load_encoder(self, state_dict):
        self.encoder.load_state_dict(state_dict)
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.encoder.eval()
        print(f"Encoder Model loaded")

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

    def forward(self, x1, x2):
        pred_flow = None
        self.update.self_attn = True
        net, inp = self.context_net(x1)  # extract context from x1

        f1, x1_skips = self.encoder(x1)
        f2, x2_skips = self.encoder(x2)
        x1_skips[16] = f1
        x2_skips[16] = f2

        # keys = list(x1_skips.keys())
        # keys.reverse()

        feature1 = list(map(x1_skips.get, self.keys))
        feature2 = list(map(x2_skips.get, self.keys))
        net1 = list(map(net.get, self.keys))
        inp1 = list(map(inp.get, self.keys))

        pred_flow = self.update(feature1, feature2, net1, inp1, pred_flow)

        # for key in keys:
        #     feature1 = x1_skips[key]
        #     feature2 = x2_skips[key]
        #     if key == keys[-1]:
        #         upsample = False
        #
        #     pred_flow = self.update(pred_flow, feature1, feature2, net[key], inp[key], key, upsample=upsample)

        return pred_flow
