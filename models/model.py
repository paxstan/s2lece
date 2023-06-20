import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.featurenet import FeatureExtractorNet, ContextNet
from models.correlationnet import CorrelationNet
from models.attention import SelfAttention, CrossAttention
from models.update import BasicUpdateBlock
from models.utils import correlate, warp, linear_position_embedding_sine, de_conv_layer, Correlation1D, \
    PositionEmbeddingSine, coords_grid


class FlowModel(nn.Module):
    def __init__(self, config, device, train=True):
        super(FlowModel, self).__init__()
        self.config = config
        self.device = device
        self.to(self.device)
        self.feature_net = FeatureExtractorNet().to(self.device)
        if train:
            self.load_encoder()
        self.patch_size = [4, 8, 8, 16, 16, 32]
        self.dilation_patch = [2, 4, 4, 8, 8, 16]

    def calculate_n_parameters(self):
        def times(shape):
            parameters = 1
            for layer in list(shape):
                parameters *= layer
            return parameters

        layer_params = [times(x.size()) for x in list(self.parameters())]

        return sum(layer_params)

    def load_encoder(self):
        if os.path.exists(self.config['fe_save_path']):
            fe_net_weights = torch.load(self.config['fe_save_path'])
            self.feature_net.load_state_dict(fe_net_weights["state_dict"])
            for params in self.feature_net.parameters():
                params.requires_grad = False
            print(f"AE Model loaded from {self.config['fe_save_path']}")
        else:
            print(f"AE Model is not in the path {self.config['fe_save_path']}")

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
        x1_fe_out = [x1]
        x2_fe_out = [x2]

        for layers in self.feature_net.encoder:
            x1 = layers(x1)
            x2 = layers(x2)
            x1_fe_out.append(x1)
            x2_fe_out.append(x2)

        return x1_fe_out, x2_fe_out


class FlowCorrelationCNN(FlowModel):
    def __init__(self, config, device, train=True):
        super(FlowCorrelationCNN, self).__init__(config, device, train)
        self.correlation_net = CorrelationNet().to(self.device)

    def forward(self, x1, x2):
        predict_flow = None
        x1_fe_out, x2_fe_out = self.extract_feature(x1, x2)

        i = len(x1_fe_out) - 1
        j = 0
        wrapped_x2_out = x2_fe_out[i]

        for layers in self.correlation_net.correlation:
            b, c, h, w = wrapped_x2_out.size()
            flow = torch.zeros((b, 2, h, w))
            if wrapped_x2_out.is_cuda:
                flow.contiguous().cuda(non_blocking=True)
            # corr_val = self.corr(x1_fe_out[i], wrapped_x2_out)
            x1_x2_correlate = correlate(x1_fe_out[i], wrapped_x2_out,
                                        patch_size=self.patch_size[j])  # [1, 16, 4, 128]

            # corr_h_mean = torch.mean(x1_x2_correlate, dim=1)
            # corr_w_mean = torch.mean(x1_x2_correlate, dim=2)
            #
            # u_disp = torch.argmax(corr_h_mean, dim=1)
            # v_disp = torch.argmax(corr_w_mean, dim=1)
            # flow[:, 0, :, :] = u_disp
            # flow[:, 1, :, :] = v_disp

            # flow_img = flow_to_color(flow.detach().squeeze().numpy().transpose(1, 2, 0))

            x1_corr_concat = torch.cat([x1_fe_out[i], x1_x2_correlate], dim=1)  # [1, 272, 4, 128]

            predict_flow = layers(x1_corr_concat)
            # flow_img_4 = flow_to_color(self.predict_flow.detach().squeeze().numpy().transpose(1, 2, 0))
            if i != 0:
                wrapped_x2_out = warp(x2_fe_out[i - 1], predict_flow)
                i = i - 1
            j = j + 1
        return predict_flow


class SleceNet(FlowModel):
    def __init__(self, config, device, train=True, embedded_dim=32, downsample_factor=8, iters=10):
        super().__init__(config, device, train)
        self.embedded_dim = embedded_dim
        self.downsample_factor = downsample_factor
        self.iters = iters
        self.context_net = ContextNet()

        self.self_attention_layer = SelfAttention(input_dim=self.embedded_dim)
        self.cross_attention_layer = CrossAttention(input_dim=self.embedded_dim)

        # Update block
        self.update_block = BasicUpdateBlock(corr_channels=self.embedded_dim,
                                             hidden_dim=self.embedded_dim,
                                             context_dim=self.embedded_dim,
                                             downsample_factor=self.downsample_factor)

    def initialize_flow(self, img, downsample=None):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h, w // downsample_factor).to(img.device)
        coords1 = coords_grid(n, h, w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def learned_upflow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, 1, self.downsample_factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.downsample_factor * flow, kernel_size=3, padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, h, self.downsample_factor * w)

    def extract_context(self, x):
        for layers in self.context_net.encoder:
            x = layers(x)
        return x

    def forward(self, x1, x2, flow_init=None):
        x1_context = self.extract_context(x1)  # extract context from x1
        net = torch.tanh(x1_context)
        inp = torch.relu(x1_context)

        x1_fe_out, x2_fe_out = self.extract_feature(x1, x2)  # extract features from x1 and x2

        feature1 = x1_fe_out[5]
        feature2 = x2_fe_out[5]
        b, dim, h, w = feature1.size()

        coords0, coords1 = self.initialize_flow(x1)

        if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + flow_init

        flow_predictions = []

        for i in range(self.iters):
            coords1 = coords1.detach()  # stop gradient
            flow = coords1 - coords0

            feature2_attn = self.cross_attention_layer(feature1, feature2)
            b, h1, w1, = feature2_attn.size()
            feature2_attn = feature2_attn.permute(0, 2, 1).view(b, w1, h, w)

            corr_val = self.corr(feature1, feature2_attn)
            b, h1, w1, = corr_val.size()

            corr_attention = self.self_attention_layer(corr_val)
            corr_attention = corr_attention.permute(0, 2, 1).view(b, self.embedded_dim, h, w)

            net, up_mask, delta_flow = self.update_block(net, inp, corr_attention, flow, upsample=True)
            coords1 = coords1 + delta_flow

            flow_up = self.learned_upflow(coords1 - coords0, up_mask)
            # x_img = flow_up[:, :, 0].reshape(-1).astype(int)
            # y_img = flow_up[:, :, 1].reshape(-1).astype(int)
            # mask_valid_in_2 = torch.zeros_like(flow_up)
            # mask_in_bound = (y_img >= 0) * (y_img < h) * (x_img >= 0) * (x_img < w)
            # mask_valid_in_2[mask_in_bound] = mask2[y_img[mask_in_bound], x_img[mask_in_bound]]
            flow_predictions.append(flow_up)

        return flow_predictions
