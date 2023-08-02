import torch.nn as nn
import torch.nn.functional as F
import torch
from spatial_correlation_sampler import spatial_correlation_sample
import math
import os


def coords_grid(batch, ht, wd, normalize=False):
    if normalize:  # [-1, 1]
        coords = torch.meshgrid(2 * torch.arange(ht) / (ht - 1) - 1,
                                2 * torch.arange(wd) / (wd - 1) - 1)
    else:
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)  # [B, 2, H, W]


def conv_layer(name, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), max_pooling=True, instance_norm=False):
    seq_model = nn.Sequential()
    seq_model.add_module(f"conv_{name}",
                         nn.Conv2d(in_channel, out_channel,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2), bias=False))
    if instance_norm:
        seq_model.add_module(f"instance_{name}", nn.InstanceNorm2d(out_channel))
    else:
        seq_model.add_module(f"bn_{name}", nn.BatchNorm2d(out_channel))
    if max_pooling:
        seq_model.add_module(f"max_pool_{name}", nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
    seq_model.add_module(f"activ_{name}", nn.LeakyReLU(0.1, inplace=True))
    return seq_model


def de_conv_layer(name, in_channel, out_channel, kernel_size=(3, 3), max_pooled=True, decoder=False):
    seq_model = nn.Sequential()
    if max_pooled:
        seq_model.add_module(f"de_conv_{name}",
                             nn.ConvTranspose2d(in_channel, out_channel,
                                                kernel_size=kernel_size, stride=(1, 2),
                                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                                output_padding=(0, 1),
                                                bias=False))
    else:
        seq_model.add_module(f"de_conv_{name}",
                             nn.ConvTranspose2d(in_channel, out_channel,
                                                kernel_size=kernel_size, stride=1,
                                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                                bias=False))
    if decoder:
        seq_model.add_module(f"de_bn_{name}", nn.BatchNorm2d(out_channel))
        seq_model.add_module(f"de_activ_{name}", nn.LeakyReLU(0.1, inplace=True))
    return seq_model


def predict_flow_layer(in_planes):
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
    # out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
    # visualize_correlation(out_corr, ph)
    out_corr = out_corr / input1.size(1)
    return F.leaky_relu(out_corr, 0.01)


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
    # mask = torch.autograd.Variable(torch.ones(x.size()))

    if x.is_cuda:
        grid = grid.cuda()
        # mask = mask.cuda()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)

    mask = nn.functional.grid_sample(torch.ones_like(x), vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def up_sample_flow(flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8 * H, 8 * W)


def linear_position_embedding_sine(x, dim=128, NORMALIZE_FACOR=1 / 200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    b, dim, f = x.size()
    frequency = torch.linspace(0, dim, dim).to(x.device)
    pos_tensor = torch.ones_like(x)
    for i in range(dim):
        if i % 2 == 0:
            pos_tensor[:, i] = torch.sin(x[:, i] * (i + 1) * pos_tensor[:, i] * math.pi)
        else:
            pos_tensor[:, i] = torch.cos(x[:, i] * (i + 1) * pos_tensor[:, i] * math.pi)

    # pos_em1 = torch.sin(math.pi * x[..., -2:-1] * frequency) + torch.cos(
    #     math.pi * x[..., -2:-1] * frequency)
    # pose_em2 = torch.sin(math.pi * x[..., -1:] * frequency) + torch.cos(math.pi * x[..., -1:] * frequency)
    # pos_embed = torch.cat([
    #     torch.sin(frequency * math.pi * x[..., -2:-1]) + torch.cos(frequency * math.pi * x[..., -2:-1]),
    #     torch.sin(frequency * math.pi * x[..., -1:]) + torch.cos(frequency * math.pi * x[..., -1:])
    # ], dim=-1)
    return pos_tensor
    # freq_bands = torch.linspace(0, dim, dim).to(x.device)  # (0, 127, 128)
    # return torch.cat([
    #     torch.sin(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
    #     torch.cos(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
    #     torch.sin(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR),
    #     torch.cos(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR)],
    #     dim=-1)


def exp_position_embedding_sine(x, dim=128, NORMALIZE_FACOR=1 / 200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    freq_bands = torch.linspace(0, dim // 4 - 1, dim // 4).to(x.device)
    return torch.cat([torch.sin(x[..., -2:-1] * (NORMALIZE_FACOR * 2 ** freq_bands)),
                      torch.cos(x[..., -2:-1] * (NORMALIZE_FACOR * 2 ** freq_bands)),
                      torch.sin(x[..., -1:] * (NORMALIZE_FACOR * 2 ** freq_bands)),
                      torch.cos(x[..., -1:] * (NORMALIZE_FACOR * 2 ** freq_bands))], dim=-1)


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=512, sine_type='lin_sine'):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i + 1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2 ** i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat(
            [torch.sin(i * math.pi * inputs) for i in self.bases] + [torch.cos(i * math.pi * inputs) for i in
                                                                     self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out


class Correlation1D:
    def __init__(self, feature1, feature2,
                 radius=32,
                 x_correlation=False,
                 ):
        self.radius = radius
        self.x_correlation = x_correlation

        if self.x_correlation:
            self.corr = self.corr_x(feature1, feature2)  # [B*H*W, 1, 1, W]
        else:
            self.corr = self.corr_y(feature1, feature2)  # [B*H*W, 1, H, 1]

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
        b, h, w = coords.shape[:3]

        if self.x_correlation:
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.zeros_like(dx)
            delta_x = torch.stack((dx, dy), dim=-1).to(coords.device)  # [2r+1, 2]

            coords_x = coords[:, :, :, 0]  # [B, H, W]
            coords_x = torch.stack((coords_x, torch.zeros_like(coords_x)), dim=-1)  # [B, H, W, 2]

            centroid_x = coords_x.view(b * h * w, 1, 1, 2)  # [B*H*W, 1, 1, 2]
            coords_x = centroid_x + delta_x  # [B*H*W, 1, 2r+1, 2]

            coords_x = 2 * coords_x / (w - 1) - 1  # [-1, 1], y is always 0

            corr_x = F.grid_sample(self.corr, coords_x, mode='bilinear',
                                   align_corners=True)  # [B*H*W, G, 1, 2r+1]

            corr_x = corr_x.view(b, h, w, -1)  # [B, H, W, (2r+1)*G]
            corr_x = corr_x.permute(0, 3, 1, 2).contiguous()  # [B, (2r+1)*G, H, W]
            return corr_x
        else:  # y correlation
            dy = torch.linspace(-r, r, 2 * r + 1)
            dx = torch.zeros_like(dy)
            delta_y = torch.stack((dx, dy), dim=-1).to(coords.device)  # [2r+1, 2]
            delta_y = delta_y.unsqueeze(1).unsqueeze(0)  # [1, 2r+1, 1, 2]

            coords_y = coords[:, :, :, 1]  # [B, H, W]
            coords_y = torch.stack((torch.zeros_like(coords_y), coords_y), dim=-1)  # [B, H, W, 2]

            centroid_y = coords_y.view(b * h * w, 1, 1, 2)  # [B*H*W, 1, 1, 2]
            coords_y = centroid_y + delta_y  # [B*H*W, 2r+1, 1, 2]

            coords_y = 2 * coords_y / (h - 1) - 1  # [-1, 1], x is always 0

            corr_y = F.grid_sample(self.corr, coords_y, mode='bilinear',
                                   align_corners=True)  # [B*H*W, G, 2r+1, 1]

            corr_y = corr_y.view(b, h, w, -1)  # [B, H, W, (2r+1)*G]
            corr_y = corr_y.permute(0, 3, 1, 2).contiguous()  # [B, (2r+1)*G, H, W]

            return corr_y

    def corr_x(self, feature1, feature2):
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c ** 0.5

        # x direction
        feature1 = feature1.permute(0, 2, 3, 1)  # [B, H, W, C]
        feature2 = feature2.permute(0, 2, 1, 3)  # [B, H, C, W]
        corr = torch.matmul(feature1, feature2)  # [B, H, W, W]

        corr = corr.unsqueeze(3).unsqueeze(3)  # [B, H, W, 1, 1, W]
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)  # [B*H*W, 1, 1, W]

        return corr

    def corr_y(self, feature1, feature2):
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c ** 0.5

        # y direction
        feature1 = feature1.permute(0, 3, 2, 1)  # [B, W, H, C]
        feature2 = feature2.permute(0, 3, 1, 2)  # [B, W, C, H]
        corr = torch.matmul(feature1, feature2)  # [B, W, H, H]

        corr = corr.permute(0, 2, 1, 3).contiguous().view(b, h, w, 1, h, 1)  # [B, H, W, 1, H, 1]
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)  # [B*H*W, 1, H, 1]

        return corr


class PositionEmbeddingSine(nn.Module):
    """
    https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def ae_loss_criterion(pred_img, img, mask):
    squared_diff = (pred_img - img) ** 2
    squared_mask = squared_diff * mask
    loss = torch.sum(squared_mask) / torch.sum(mask)
    return loss


def s2lece_loss_criterion(flow_pred, flow_gt, train=True):
    # ae = AE()
    """ Loss function defined over sequence of flow predictions
     """

    if not train:
        torch.set_grad_enabled(False)

    flow_loss = torch.tensor(0.0, dtype=torch.float32).to(flow_gt.device)
    # gt_flow = torch.floor(initial_flow + flow_gt).to(torch.int32)
    # valid_flow = ((gt_flow[:, 0] >= 0) & (gt_flow[:, 0] < 32)) & ((gt_flow[:, 1] >= 0) & (gt_flow[:, 1] < 2000))
    #
    # valid_flow = valid_flow[:, None]

    # valid_flow *= mask

    # i_loss = (flow_pred - flow_gt).abs()
    # flow_loss += (valid_flow * i_loss).mean()

    mask = (flow_gt != 0).float()

    # squared_diff = (flow_pred - flow_gt) ** 2
    #
    # masked_square_diff = squared_diff * mask
    #
    # mse_loss = torch.sum(masked_square_diff) / torch.sum(mask)

    rmse_loss = rmse_loss_fn(flow_pred, flow_gt, mask)

    aae_loss = average_angular_error_fn(flow_pred, flow_gt, mask)

    div_curl_loss = div_curl_loss_fn(flow_pred, flow_gt, mask, div_weight=1, curl_weight=1)

    # gt_flow = flow_gt * valid_flow
    # pred_flow = flow_pred * valid_flow

    flow_loss += (
            torch.tensor(rmse_loss.clone().detach(), requires_grad=True if train else False) +
            torch.tensor(aae_loss.clone().detach(), requires_grad=True if train else False) +
            torch.tensor(div_curl_loss.clone().detach(), requires_grad=True if train else False))

    # print(flow_loss)

    diff = (flow_pred - flow_gt).abs()

    masked_diff = diff * mask

    mae_loss = torch.sum(masked_diff) / torch.sum(mask)

    epe = torch.sum((flow_pred * mask - flow_gt * mask) ** 2, dim=1).sqrt()
    # epe = epe * mask

    metrics = {
        'mae': mae_loss,
        'epe': torch.sum(epe) / torch.sum(mask),
        '1px': torch.sum((epe > 1).float()) / torch.sum(mask),
        '3px': torch.sum((epe > 3).float()) / torch.sum(mask),
        '5px': torch.sum((epe > 5).float()) / torch.sum(mask)
    }

    return flow_loss, metrics


def compute_divergence(flow):
    """Compute the divergence of the flow field."""
    # Compute gradients along the spatial dimensions (H and W) of the flow field
    dU_dx = flow[:, 0].unsqueeze(1).contiguous()
    dV_dy = flow[:, 1].unsqueeze(1).contiguous()

    # Compute the divergence as the sum of the gradients
    divergence = dU_dx + dV_dy
    return divergence


def compute_curl(flow):
    """Compute the curl of the flow field."""
    # Compute gradients along the spatial dimensions (H and W) of the flow field
    dU_dy = flow[:, 0].unsqueeze(1).contiguous()
    dV_dx = flow[:, 1].unsqueeze(1).contiguous()

    # Compute the curl as the difference between the gradients
    curl = dV_dx - dU_dy
    return curl


def rmse_loss_fn(pred_flow, gt_flow, mask):
    squared_diff = (pred_flow - gt_flow) ** 2

    rse_loss = torch.sqrt(squared_diff)

    masked_rse_loss = rse_loss * mask

    rmse_loss = torch.sum(masked_rse_loss) / torch.sum(mask)

    # rmse_loss = torch.sqrt(mse_loss)

    return rmse_loss


def average_angular_error_fn(pred_flow, gt_flow, mask):
    """Calculate the Average Angular Error (AAE) between predicted and ground truth flows."""

    y_pred = pred_flow * mask
    y_true = gt_flow * mask
    dotP = torch.sum(y_pred * y_true, dim=1) + 1
    Norm_pred = torch.sqrt(torch.sum(y_pred * y_pred, dim=1) + 1)
    Norm_true = torch.sqrt(torch.sum(y_true * y_true, dim=1) + 1)
    # ae = 180 / math.pi * torch.acos(dotP / (Norm_pred * Norm_true))
    ae = 180 / math.pi * torch.acos(torch.clamp((dotP / (Norm_pred * Norm_true)), -1.0 + 1e-8, 1.0 - 1e-8))

    # ae_mean = torch.sum(ae) / torch.sum(mask)
    # return ae.mean(1).mean(1)
    mean = ae.mean()

    if torch.isnan(mean):
        print("nan!!!!")
        mean = torch.tensor(0.0)
    return mean


def div_curl_loss_fn(pred_flow, gt_flow, mask, div_weight=0.8, curl_weight=0.2):
    div = compute_divergence(gt_flow)
    curl = compute_curl(gt_flow)

    div_hat = compute_divergence(pred_flow)
    curl_hat = compute_curl(pred_flow)

    div_norm = torch.norm(div * mask - div_hat * mask)
    curl_norm = torch.norm(curl * mask - curl_hat * mask)

    div_curl = (div_weight * div_norm) + (curl_weight * curl_norm)

    div_curl_loss = torch.sum(div_curl) / torch.sum(mask)

    # divergence_loss = torch.mean((div_hat * mask - div * mask).abs())
    # curl_loss = torch.mean((curl_hat * mask - curl * mask).abs())
    #
    # # Combine the divergence and curl losses
    # total_loss = divergence_loss + curl_loss

    return div_curl_loss


def abs_flow_builder(initial_flow, pred_flow, idx1, idx2):
    abs_flow = torch.floor(initial_flow + pred_flow)
    x_img = abs_flow[0].astype(int)
    y_img = abs_flow[1].astype(int)

    invalid_x = torch.where(x_img >= 32)
    invalid_y = torch.where(y_img >= 2000)

    x_img[invalid_x] = invalid_x[0]
    y_img[invalid_y] = invalid_y[1]

    x_img = x_img.reshape(-1)
    y_img = y_img.reshape(-1)
    corres_idx2 = (idx2[x_img.astype(int), y_img.astype(int)]).reshape(-1, 1)

    corres_id = torch.cat((idx1, corres_idx2))

    valid_index = (corres_id[:, 0] != -1) & (corres_id[:, 1] != -1)
    valid_corres_id = corres_id[valid_index]
    # b1, c1, h1, w1 = range1.size()
    # b2, h2, w2 = mask2.size()
    # b, c, h, w = flow.shape
    # flow = flow.permute(1, 0, 2, 3)
    # abs_flow = torch.zeros_like(flow)
    # abs_flow[0, :] = torch.arange(h).unsqueeze(1)
    # abs_flow[1, :] = torch.arange(w)
    #
    # # Perform floor operation and add pred_flow
    # abs_flow = torch.floor(abs_flow + flow).int()
    #
    # range1 = range1.reshape(-1)
    # range2 = range2.reshape(-1)
    # mask2 = mask2.view(b1*h2, w2).bool()
    # # abs_flow = abs_flow.view(b1, 2, -1).int()
    # x_img = abs_flow[0].reshape(-1)
    # y_img = abs_flow[1].reshape(-1)
    # mask_valid_in_2 = torch.zeros((b1 * h1 * w1), dtype=torch.bool).to(mask2.device)
    # mask_in_bound = (x_img >= 0) * (x_img < h2) * (y_img >= 0) * (y_img < w2)
    # # y_img = y_img[mask_in_bound]
    # mask_valid_in_2[mask_in_bound] = mask2[x_img[mask_in_bound], y_img[mask_in_bound]]
    #
    # # get ranges of valid image2 points
    # r2 = range2[mask_valid_in_2]
    # # get corresponding transformed ranges of image1
    # r1_t = range1[mask_valid_in_2]
    #
    # # check if points in image1 are occluded in 2
    # d = (r2 - r1_t) / r1_t
    # # 20% threshold when close to each other
    # not_occluded = d > -0.2
    # # # define occlusion mask
    # occlusion_mask = torch.ones((b1* h1 * w1), dtype=torch.bool).to(mask2.device)
    # occlusion_mask[mask_valid_in_2] = not_occluded
    #
    # # set mask_valid_in_2 false when occluded
    # mask_valid_in_2 = mask_valid_in_2 & occlusion_mask
    # mask_valid_in_2 = mask_valid_in_2.view(b1, h1, w1)
    #
    # # mask flow according to occluded points
    # flow[0][~mask_valid_in_2] = 0
    # flow[1][~mask_valid_in_2] = 0

    flow = flow.permute(1, 0, 2, 3)

    return flow, mask_valid_in_2


def load_encoder(feature_net, ae_path):
    if os.path.exists(ae_path):
        fe_net_weights = torch.load(ae_path)
        feature_net.load_state_dict(fe_net_weights["state_dict"])
        for params in feature_net.parameters():
            params.requires_grad = False
        print(f"AE Model loaded from {ae_path}")
    else:
        print(f"AE Model is not in the path {ae_path}")
    return feature_net.encoder
