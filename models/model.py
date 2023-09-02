import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.featurenet import FeatureExtractorNet
from models.update import BasicUpdateBlock
from models.model_utils import coords_grid


class UpdateNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.update_block = BasicUpdateBlock(corr_channels=243,
                                             hidden_dim=256,
                                             downsample_factor=16)

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
    def learned_upflow(flow, mask, downsample):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, 1, downsample, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(downsample * flow, kernel_size=3, padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, h, downsample * w)

    def forward(self, feature1, feature2, net1, inp1, mask1, iters, flow_init=None):
        coords0, coords1 = self.initialize_flow(feature1[0])
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords0 = coords0.detach()
            coords1 = coords1.detach()  # stop gradient
            flow = coords1 - coords0

            corr_fn = CorrBlock(feature1[0], feature2[0], radius=4)
            corr = corr_fn(coords1)
            net, up_mask, delta_flow = self.update_block(net1[0], inp1[0], corr, flow, upsample=True)
            coords1 = coords1 + delta_flow
            flow_up = self.learned_upflow(coords1 - coords0, up_mask, downsample=16)
            flow_up *= mask1
            flow_predictions.append(flow_up)
        return flow_predictions


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
    def __init__(self, iters, fe_params, update_params):
        super().__init__()
        self.iters = iters
        self.keys = [16]
        self.encoder = FeatureExtractorNet(fe_params)
        fe_params["type"] = "cn"
        self.context_net = FeatureExtractorNet(fe_params)
        self.update = UpdateNet(update_params)

    def load_encoder(self, state_dict):
        self.encoder.load_state_dict(state_dict)
        for params in self.encoder.parameters():
            params.requires_grad = False
        # Unfreeze the parameters of the last layer
        for param in self.encoder.enc4.parameters():
            param.requires_grad = True  # Unfreeze the last layer's parameters
        # self.encoder.eval()
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

    def forward(self, x1, x2, mask1, pred_flow=None):
        f1, x1_skips = self.encoder(x1)
        f2, x2_skips = self.encoder(x2)
        c1, c1_skips = self.context_net(x1)
        x1_skips[16] = f1
        x2_skips[16] = f2
        c1_skips[16] = c1

        net1 = []
        inp1 = []

        feature1 = list(map(x1_skips.get, self.keys))
        feature2 = list(map(x2_skips.get, self.keys))
        context1 = list(map(c1_skips.get, self.keys))
        for idx, value in enumerate(context1):
            net1.append(torch.tanh(value))
            inp1.append(torch.relu(value))

        # feature1 = [f1]
        # feature2 = [f2]
        # net1 = [torch.tanh(c1)]
        # inp1 = [torch.relu(c1)]

        pred_flow = self.update(feature1, feature2, net1, inp1, mask1, iters=self.iters, flow_init=pred_flow)
        return pred_flow
