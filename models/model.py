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
        # self.self_attention = SelfAttention(embed_dim=embedded_dim, in_channel=4000)

        # self.cross_attention_16 = CrossAttention(embed_dim=embedded_dim,
        #                                          in_channel_source=32, in_channel_target=32)
        # self.update_block_16 = BasicUpdateBlock(corr_channels=32,
        #                                         hidden_dim=embedded_dim,
        #                                         context_dim=32,
        #                                         downsample_factor=4)

        # self.update_block_32 = BasicUpdateBlock(corr_channels=27,
        #                                         hidden_dim=32,
        #                                         downsample_factor=4)

        self.update_block_16 = BasicUpdateBlock(corr_channels=243,
                                                hidden_dim=16,
                                                downsample_factor=16)  # 4

        # self.update_block_8 = BasicUpdateBlock(corr_channels=243,
        #                                        hidden_dim=8,
        #                                        downsample_factor=8)  # 4
        #
        # self.update_block_4 = BasicUpdateBlock(corr_channels=243,
        #                                        hidden_dim=4,
        #                                        downsample_factor=4)
        #
        # self.update_block_1 = BasicUpdateBlock(corr_channels=243,
        #                                        hidden_dim=1,
        #                                        downsample_factor=4, learn_upsample=False)

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
        # print("key 16")
        coords0, coords1 = self.initialize_flow(feature1[0])
        if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + flow_init

        coords0 = coords0.detach()
        coords1 = coords1.detach()  # stop gradient
        flow = coords1 - coords0

        corr_fn = CorrBlock(feature1[0], feature2[0], radius=1)
        corr = corr_fn(coords1)
        # corr_val = self.corr(feature1[0], feature2[0])
        # corr_attn = self.self_attention(feature1[0].size(), corr_val)
        # corr_attn = corr_attn.permute(0, 2, 1).view(feature1[0].size())

        # net, up_mask, delta_flow = self.update_block_16(net1[0], inp1[0], corr_attn, flow, upsample=True)
        net, up_mask, delta_flow = self.update_block_32(net1[0], inp1[0], corr, flow, upsample=True)
        coords1 = coords1 + delta_flow
        flow = coords1 - coords0
        # flow, corr_attn = self.learned_upflow(flow, corr_attn, up_mask, downsample=4)
        flow, corr = self.learned_upflow(flow, corr, up_mask, downsample=32)

        # key 16
        # # print("key 16")
        # coords0, coords1 = self.initialize_flow(feature1[1])
        # if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
        #     coords1 = coords1 + flow_init
        #
        # coords0 = coords0.detach()
        # coords1 = coords1.detach()  # stop gradient
        # flow = coords1 - coords0
        #
        # corr_fn = CorrBlock(feature1[1], feature2[1], radius=4)
        # corr = corr_fn(coords1)
        # # corr_val = self.corr(feature1[0], feature2[0])
        # # corr_attn = self.self_attention(feature1[0].size(), corr_val)
        # # corr_attn = corr_attn.permute(0, 2, 1).view(feature1[0].size())
        #
        # # net, up_mask, delta_flow = self.update_block_16(net1[0], inp1[0], corr_attn, flow, upsample=True)
        # net, up_mask, delta_flow = self.update_block_16(net1[1], inp1[1], corr, flow, upsample=True)
        # coords1 = coords1 + delta_flow
        # flow = coords1 - coords0
        # # flow, corr_attn = self.learned_upflow(flow, corr_attn, up_mask, downsample=4)
        # flow, corr = self.learned_upflow(flow, corr, up_mask, downsample=16)

        # key 8
        # coords0, coords1 = self.initialize_flow(feature1[1])
        #
        # coords1 = coords1 + flow
        # coords0 = coords0.detach()
        # coords1 = coords1.detach()  # stop gradient
        # flow = coords1 - coords0
        #
        # corr_fn = CorrBlock(feature1[1], feature2[1], radius=4)
        # corr = corr_fn(coords1)
        # # corr_val = self.corr(feature1[0], feature2[0])
        # # corr_attn = self.self_attention(feature1[0].size(), corr_val)
        # # corr_attn = corr_attn.permute(0, 2, 1).view(feature1[0].size())
        #
        # # net, up_mask, delta_flow = self.update_block_16(net1[0], inp1[0], corr_attn, flow, upsample=True)
        # net, up_mask, delta_flow = self.update_block_8(net1[1], inp1[1], corr, flow, upsample=True)
        # coords1 = coords1 + delta_flow
        # flow = coords1 - coords0
        # # flow, corr_attn = self.learned_upflow(flow, corr_attn, up_mask, downsample=4)
        # flow, corr = self.learned_upflow(flow, corr, up_mask, downsample=8)

        # key 4
        # print("key 4")
        # _, _, h, w = feature1[1].size()
        # coords0, coords1 = self.initialize_flow(feature1[1])
        #
        # coords1 = coords1 + flow
        #
        # # coords0 = coords0.detach()
        # # coords1 = coords1.detach()  # stop gradient
        # # flow = coords1 - coords0
        #
        # # corr_attn = self.cross_attention_4(corr_attn, feature1[1])
        # # b, h1, w1, = corr_attn.size()
        # # corr_attn = corr_attn.permute(0, 2, 1).view(b, w1, h, w)
        #
        # corr_fn = CorrBlock(feature1[1], feature2[1], radius=4)
        # corr = corr_fn(coords1)
        # net, up_mask, delta_flow = self.update_block_4(net1[1], inp1[1], corr, flow, upsample=True)
        # coords1 = coords1 + delta_flow
        # flow = coords1 - coords0
        # flow, corr = self.learned_upflow(flow, corr, up_mask, downsample=4)
        #
        # # key 1
        # #
        # print("key 1")
        # _, _, h, w = feature1[2].size()
        # coords0, coords1 = self.initialize_flow(feature1[2])
        #
        # coords1 = coords1 + flow
        # # corr_attn = self.cross_attention_1(corr_attn, feature1[2])
        # # b, h1, w1, = corr_attn.size()
        # # corr_attn = corr_attn.permute(0, 2, 1).view(b, w1, h, w)
        #
        # net, up_mask, delta_flow = self.update_block_1(net1[2], inp1[2], corr, flow, upsample=False)
        # coords1 = coords1 + delta_flow
        # flow = coords1 - coords0

        return flow


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=3, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=(1, 2))
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


class S2leceNet(nn.Module):
    def __init__(self, config, fe_params, update_params):
        super().__init__()
        self.config = config
        self.keys = [32, 8]
        # self.context_net = ContextNet().float()
        self.encoder = FeatureExtractorNet(fe_params)
        fe_params["type"] = "cn"
        self.context_net = FeatureExtractorNet(fe_params)
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
        # net, inp = self.context_net(x1)  # extract context from x1

        f1, x1_skips = self.encoder(x1)
        f2, x2_skips = self.encoder(x2)
        c1, c1_skips = self.context_net(x1)
        x1_skips[32] = f1
        x2_skips[32] = f2
        c1_skips[32] = c1

        net1 = []
        inp1 = []

        feature1 = list(map(x1_skips.get, self.keys))
        feature2 = list(map(x2_skips.get, self.keys))
        context1 = list(map(c1_skips.get, self.keys))
        for idx, value in enumerate(context1):
            net1.append(torch.tanh(value))
            inp1.append(torch.relu(value))

        pred_flow = self.update(feature1, feature2, net1, inp1, pred_flow)
        return pred_flow
