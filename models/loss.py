import torch
import torch.nn.functional as F
from utils import pytorch_ssim
from models.model_utils import warp
import math

if hasattr(F, 'interpolate'):
    interpolate = torch.nn.functional.interpolate


def rmse_loss_fn(pred_flow, target_flow, sparse=True):
    """Function to calculate Root Mean Square Error (RMSE) between predicted and ground
    truth flows(sparsity inclusion)"""
    squared_diff = (pred_flow - target_flow) ** 2

    rse_loss = torch.sqrt(squared_diff)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        rse_loss = rse_loss.permute(0, 2, 3, 1)

        if not mask.all():
            rse_loss = rse_loss[~mask]

    rmse_loss = rse_loss.mean()

    if torch.isnan(rmse_loss):
        rmse_loss = torch.tensor(0.0).to(target_flow.device)
    return rmse_loss


def average_angular_error_fn(pred_flow, target_flow, sparse=True):
    """Function to calculate Average Angular Error (AAE) between predicted and ground truth flows(sparsity inclusion)"""

    dotP = torch.sum(pred_flow * target_flow, dim=1)
    Norm_pred = torch.sqrt(torch.sum(pred_flow * pred_flow, dim=1))
    Norm_true = torch.sqrt(torch.sum(target_flow * target_flow, dim=1))
    ae = 180 / math.pi * torch.acos(dotP / (Norm_pred * Norm_true + 1e-6))

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        if not mask.all():
            ae = ae[~mask]

    mean = ae.mean()
    if torch.isnan(mean):
        mean = torch.tensor(0.0).to(target_flow.device)
    return mean


def epe_loss_fn(input_flow, target_flow, sparse=True, mean=True):
    """Function to calculate End-Point-Error (EPE) between predicted and ground truth flows(sparsity inclusion)"""
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)

    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        if not mask.all():
            EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def charbonnier_penalty(x, e=1e-8, delta=0.4, averge=True):
    """Function to calculate Charbonnier Penalty between predicted and ground image of Autoencoder"""
    p = ((x) ** 2 + e).pow(delta)
    if averge:
        p = p.mean()
    else:
        p = p.sum()
    return p


def patch_mse_loss(input_images, target_images, mask=None, patch_size=16, step=16):
    """Function to calculate Mean Square Error (Patch wise) between predicted and ground image of Autoencoder"""
    # Ensure the input images have the same shape
    assert input_images.shape == target_images.shape, "Input and target images must have the same shape"

    mse_masked = torch.empty(0)
    # Extract patch tensors using unfold operation
    input_patches = input_images.unfold(2, patch_size, step).unfold(3, patch_size, step)
    target_patches = target_images.unfold(2, patch_size, step).unfold(3, patch_size, step)

    # Calculate MSE loss for each patch and flatten the patches along spatial dimensions
    mse_loss = F.mse_loss(input_patches, target_patches, reduction='none')

    b, c, h1, w1, p1, p2 = mse_loss.shape
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
    """Function to calculate Mean Absolute Error between predicted and ground truth flows(sparsity inclusion)"""
    abs_loss = (pred_flow - target_flow).abs()
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        abs_loss = abs_loss.permute(0, 2, 3, 1)
        if not mask.all():
            abs_loss = abs_loss[~mask]

    return abs_loss.mean()


def flow_loss_metric(pred_flow, target_flow):
    epe_loss = epe_loss_fn(pred_flow, target_flow)
    rmse_loss = rmse_loss_fn(pred_flow, target_flow)
    aae_loss = average_angular_error_fn(pred_flow, target_flow)
    mae_loss = mae_loss_fn(pred_flow, target_flow)

    return [epe_loss, rmse_loss, aae_loss, mae_loss]


def reconstruct_loss_metric(f1, f2):
    diff_loss = charbonnier_penalty(f2 - f1, delta=0.4, averge=True)
    ssim_loss = 1.0 - pytorch_ssim.ssim(f1, f2, window_size=16, size_average=True)  # [0, 1]
    mse_loss = patch_mse_loss(f1, f2)
    psnr_loss = -10.0 * ((1.0 / (mse_loss + 1)).log10())

    return [diff_loss, ssim_loss, psnr_loss, mse_loss]


def sequence_loss(flow_preds, target_flow, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - target_flow).abs()
        i_loss = i_loss.permute(0, 2, 3, 1)
        if not mask.all():
            i_loss = i_loss[~mask]
        flow_loss += (i_weight * i_loss.mean())
    return flow_loss


def autoencoder_loss_fn(pred_img, target_image, mask):
    if mask is not None:
        pred_img = pred_img * mask
        mask_loss = (1.0 - mask).mean()
    else:
        mask_loss = 0.0
    diff_loss, ssim_loss, psnr_loss, mse_loss = reconstruct_loss_metric(target_image, pred_img)
    metrics = {
        "diff loss": diff_loss.detach(), "ssim loss": ssim_loss.detach(),
        "psnr loss": psnr_loss.detach(), "mask loss": mask_loss.detach()
    }
    return mse_loss, metrics


def flow_loss_fn(img1, img2, target_flow, pred_flow_list, mask, max_flow=400):
    assert img1.shape == img2.shape, "Input and target images must have the same shape"

    if mask is not None:
        img1 = img1 * mask
        mask_loss = (1.0 - mask).mean()
    else:
        mask_loss = 0.0

    magnitude = torch.sum(target_flow ** 2, dim=1).sqrt().unsqueeze(dim=1)
    valid_flow_mask = (magnitude < max_flow) & (mask == 1)

    target_flow = target_flow * valid_flow_mask
    pred_flow = pred_flow_list[-1]
    flow_loss = sequence_loss(pred_flow_list, target_flow)

    warped_img1 = warp(img1, pred_flow)

    epe_loss, rmse_loss, aae_loss, mae_loss = flow_loss_metric(pred_flow, target_flow)
    diff_loss, ssim_loss, psnr_loss, mse_loss = reconstruct_loss_metric(img2, warped_img1)

    metrics = {
        "flow": {
            "rmse loss": rmse_loss.detach(), "epe loss": epe_loss.detach(),
            "aae loss": aae_loss.detach(), "mae loss": mae_loss.detach()
        },
        "reconstruct": {
            "diff loss": diff_loss.detach(), "ssim loss": ssim_loss.detach(),
            "psnr loss": psnr_loss.detach(), "mse loss": mse_loss.detach(),
            "mask loss": mask_loss.detach()
        }
    }

    return flow_loss, metrics
