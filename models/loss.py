import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import pytorch_ssim
from models.model_utils import warp
import math

if hasattr(F, 'interpolate'):
    interpolate = torch.nn.functional.interpolate


def compute_divergence(flow):
    """Compute the divergence of the flow field."""
    # Compute gradients along the spatial dimensions (H and W) of the flow field
    # dU_dx = flow[:, 0].unsqueeze(1).contiguous()
    # dV_dy = flow[:, 1].unsqueeze(1).contiguous()

    dU_dx = flow[:, 0].contiguous()
    dV_dy = flow[:, 1].contiguous()

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


def rmse_loss_fn(pred_flow, target_flow, sparse=True):
    squared_diff = (pred_flow - target_flow) ** 2

    rse_loss = torch.sqrt(squared_diff)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        # print(mask.shape, rse_loss.shape)
        rse_loss = rse_loss.permute(0, 2, 3, 1)

        rse_loss = rse_loss[~mask]
        # print(mask.shape, rse_loss.shape)

    # masked_rse_loss = rse_loss * mask

    # rmse_loss = torch.sum(masked_rse_loss) / torch.sum(mask)
    rmse_loss = rse_loss.mean()

    if torch.isnan(rmse_loss):
        rmse_loss = torch.tensor(0.0)
    # rmse_loss = torch.sqrt(mse_loss)

    return rmse_loss


def average_angular_error_fn(pred_flow, target_flow, sparse=True):
    """Calculate the Average Angular Error (AAE) between predicted and ground truth flows."""

    # y_pred = pred_flow[:, :, mask.squeeze()]
    # y_true = gt_flow[:, :, mask.squeeze()]
    dotP = torch.sum(pred_flow * target_flow, dim=1)
    Norm_pred = torch.sqrt(torch.sum(pred_flow * pred_flow, dim=1))
    Norm_true = torch.sqrt(torch.sum(target_flow * target_flow, dim=1))
    ae = 180 / math.pi * torch.acos(dotP / (Norm_pred * Norm_true + 1e-6))
    # ae = 180 / math.pi * torch.acos(torch.clamp((dotP / (Norm_pred * Norm_true)), -1.0 + 1e-8, 1.0 - 1e-8))

    # ae_mean = torch.sum(ae) / torch.sum(mask)
    # return ae.mean(1).mean(1)
    # mean = ae / torch.sum(mask)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        ae = ae[~mask]

    mean = ae.mean()
    if torch.isnan(mean):
        mean = torch.tensor(0.0)
    return mean


def div_curl_loss_fn(pred_flow, target_flow, div_weight=0.8, curl_weight=0.2, sparse=True):
    # pred_flow = pred_flow * mask
    # gt_flow = gt_flow * mask
    div = compute_divergence(target_flow)
    curl = compute_curl(target_flow)

    div_hat = compute_divergence(pred_flow)
    curl_hat = compute_curl(pred_flow)

    div_norm = torch.norm(div - div_hat)
    curl_norm = torch.norm(curl - curl_hat)

    div_curl = (div_weight * div_norm) + (curl_weight * curl_norm)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        # mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        mask = (target_flow == [0, 0])

        div_curl = div_curl[~mask]

    # div_curl_loss = torch.sum(div_curl) / torch.sum(mask)

    # divergence_loss = torch.mean((div_hat * mask - div * mask).abs())
    # curl_loss = torch.mean((curl_hat * mask - curl * mask).abs())
    #
    # # Combine the divergence and curl losses
    # total_loss = divergence_loss + curl_loss
    div_curl_loss = div_curl.mean()
    if torch.isnan(div_curl_loss):
        div_curl_loss = torch.tensor(0.0)
    return div_curl_loss


def epe_loss_fn(input_flow, target_flow, sparse=True, mean=True):
    # if mask is not None:
    #     target_flow = target_flow * mask
    #     input_flow = input_flow * mask
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)

    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def charbonnier_penalty(x, e=1e-8, delta=0.4, averge=True):
    p = ((x) ** 2 + e).pow(delta)
    if averge:
        p = p.mean()
    else:
        p = p.sum()
    return p


def patch_mse_loss(input_images, target_images, mask=None, patch_size=16, step=16):
    # Ensure the input images have the same shape
    assert input_images.shape == target_images.shape, "Input and target images must have the same shape"

    mse_masked = torch.empty(0)
    # Extract patch tensors using unfold operation
    input_patches = input_images.unfold(2, patch_size, step).unfold(3, patch_size, step)
    target_patches = target_images.unfold(2, patch_size, step).unfold(3, patch_size, step)

    # Calculate MSE loss for each patch and flatten the patches along spatial dimensions
    mse_loss = F.mse_loss(input_patches, target_patches, reduction='none')

    b, c, h1, w1, p1, p2 = mse_loss.shape
    # mse_loss = mse_loss.view(mse_loss.shape[0], mse_loss.shape[1], -1).mean(dim=-1)
    mse_loss = mse_loss.reshape(b, c, h1 * w1 * p1 * p2)
    if mask is not None:
        mask_patches = mask.unfold(2, patch_size, step).unfold(3, patch_size, step)
        mask_patches = mask_patches.reshape(b, c, h1 * w1 * p1 * p2)
        mse_loss *= mask_patches
        if mask_patches.any():
            mse_masked = (torch.sum(mse_loss) / torch.sum(mask_patches))
    else:
        mse_masked = mse_loss.mean()
    mse_masked = mse_masked.to(input_images.device)
    return mse_masked


def mae_loss_fn(pred_flow, target_flow, sparse=True):
    abs_loss = (pred_flow - target_flow).abs()
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        abs_loss = abs_loss.permute(0, 2, 3, 1)

        abs_loss = abs_loss[~mask]
        # print(mask.shape, rse_loss.shape)

    return abs_loss.mean()


def flow_loss(pred_flow, target_flow):
    epe_loss = epe_loss_fn(pred_flow, target_flow)
    rmse_loss = rmse_loss_fn(pred_flow, target_flow)
    aae_loss = average_angular_error_fn(pred_flow, target_flow)
    mae_loss = mae_loss_fn(pred_flow, target_flow)
    # div_curl_loss = div_curl_loss_fn(pred_flow, target_flow, div_weight=1, curl_weight=1)

    return [epe_loss, rmse_loss, aae_loss, mae_loss]


def reconstruct_loss(f1, f2, averge=True):
    # beta_1, beta_2, beta_3 = 0.5, 0.6, 0.5
    diff_loss = charbonnier_penalty(f2 - f1, delta=0.4, averge=True)
    ssim_loss = 1.0 - pytorch_ssim.ssim(f1, f2, window_size=16, size_average=True)  # [0, 1]
    mse_loss = patch_mse_loss(f1, f2)
    psnr_loss = -10.0 * ((1.0 / (mse_loss + 1)).log10())

    # num = 1 if averge else f1.shape[-2] * f1.shape[-1]
    # diff_loss = num * beta_1 * diff_loss
    # ssim_loss = num * beta_2 * ssim_loss
    # psnr_loss = num * beta_3 * psnr_loss

    return [diff_loss, ssim_loss, psnr_loss, mse_loss]


def flow_loss_fn(img1, img2, target_flow, pred_flow, mask, max_flow=400, patch_size=16, step=16):
    assert img1.shape == img2.shape, "Input and target images must have the same shape"

    if mask is not None:
        img1 = img1 * mask
        mask_loss = (1.0 - mask).mean()
    else:
        mask_loss = 0.0

    magnitude = torch.sum(target_flow ** 2, dim=1).sqrt().unsqueeze(dim=1)
    # print((magnitude < max_flow).shape, (mask == 1).shape)
    valid_flow_mask = (magnitude < max_flow) & (mask == 1)
    # valid_flow_mask = valid_flow_mask.unsqueeze(1)

    target_flow = target_flow * valid_flow_mask
    pred_flow = pred_flow * valid_flow_mask

    warped_img1 = warp(img1, pred_flow)

    epe_loss, rmse_loss, aae_loss, mae_loss = flow_loss(pred_flow, target_flow)
    diff_loss, ssim_loss, psnr_loss, mse_loss = reconstruct_loss(img2, warped_img1)

    metrics = {
        "flow": {
            "rmse loss": rmse_loss.detach(), "epe loss": epe_loss.detach(),
            "aae loss": aae_loss.detach(),
        },
        "reconstruct": {
            "diff loss": diff_loss.detach(), "ssim loss": ssim_loss.detach(),
            "psnr loss": psnr_loss.detach(), "mse loss": mse_loss.detach(),
            "mask loss": mask_loss.detach()
        }
    }

    return mae_loss, metrics
