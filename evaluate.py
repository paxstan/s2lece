from visualization.visualization import compare_flow, visualize_point_cloud, flow_to_color
import matplotlib

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from input_pipeline.dataloader import normalize_img
import numpy as np
import torch
from models.model_utils import s2lece_loss_criterion, patch_mse_loss, ae_loss_criterion, patch_flow_loss, warp
from models.loss import flow_loss_fn
from utils import pytorch_ssim
import torch.nn.functional as F
import torch.nn as nn


def evaluate(net, img1, img2, flow=None, valid_mask=None, idx=None, metadata=None, random=False):
    img1 = normalize_img(np.array(img1))
    img2 = normalize_img(np.array(img2))
    img1 = torch.unsqueeze(torch.tensor(img1), 0)
    img2 = torch.unsqueeze(torch.tensor(img2), 0)
    mask2 = torch.unsqueeze(torch.tensor(valid_mask), 0)
    pred_flow = net(img1, img2)
    if not random:
        target_flow = torch.unsqueeze(torch.tensor(flow), 0)
        flow_loss, metrics, new_valid_mask = loss_criterion(pred_flow, target_flow, mask2, img1, img2,
                                                            train=False)
        print(flow_loss)
        print(metrics)
        compare_flow(target_flow, pred_flow, new_valid_mask, idx, flow_loss)
        visualize_point_cloud(target_flow, mask2, metadata)
        visualize_point_cloud(pred_flow[-1], new_valid_mask, metadata)
    else:
        pred_flow_masked, new_valid_mask = flow_masker(pred_flow[-1], mask2, img1, img2)
        pd_flow = pred_flow_masked.detach().squeeze().numpy()

        pred_flow_img = flow_to_color(pd_flow.transpose(1, 2, 0))
        plt.imshow(pred_flow_img)
        new_valid_mask = new_valid_mask.detach().squeeze().numpy()
        visualize_point_cloud(pred_flow_masked, new_valid_mask, metadata, transform=False)


def test_network(type_net, dataloader, net):
    ssim_loss = pytorch_ssim.SSIM(window_size=16)
    with torch.no_grad():
        if type_net == "ae":
            batch_img = torch.empty(0)
            batch_mask = torch.empty(0)
            max_count = len(dataloader)
            for idx, input_data in enumerate(dataloader):
                img = input_data.pop('img')
                mask = input_data.pop('mask')
                img = torch.unsqueeze(torch.tensor(img), 0)
                mask = torch.unsqueeze(torch.tensor(mask), 0)
                batch_img = torch.cat([batch_img, img], dim=0)
                batch_mask = torch.cat([batch_mask, mask], dim=0)
                if batch_img.shape[0] == 4:
                    pred_img = net(batch_img)
                    ssim_metric = ssim_loss(batch_img, pred_img)
                    patch_loss = patch_mse_loss(batch_img, pred_img, batch_mask)
                    mse_loss = ae_loss_criterion(pred_img, batch_img, batch_mask)
                    print(f"patch loss: {patch_loss.item()}, mse_loss: {mse_loss}, ssim: {ssim_metric}")
                    batch_img = torch.empty(0)
                    batch_mask = torch.empty(0)

                if idx == max_count - 1:
                    break

        elif type_net == "s2lece":
            batch_img1 = torch.empty(0)
            batch_img2 = torch.empty(0)
            batch_mask1 = torch.empty(0)
            batch_target_flow = torch.empty(0)
            for idx, input_data in enumerate(dataloader):
                if idx < 50:
                    img1 = input_data.pop('img1')
                    img2 = input_data.pop('img2')
                    target_flow = input_data.pop('flow')
                    mask1 = input_data.pop('mask1')
                    img1 = torch.unsqueeze(img1, 0).to(torch.float32)
                    img2 = torch.unsqueeze(img2, 0).to(torch.float32)
                    target_flow = torch.unsqueeze(torch.tensor(target_flow), 0)
                    mask1 = torch.unsqueeze(torch.tensor(mask1), 0)
                    batch_img1 = torch.cat([batch_img1, img1], dim=0)
                    batch_img2 = torch.cat([batch_img2, img2], dim=0)
                    batch_mask1 = torch.cat([batch_mask1, mask1], dim=0)
                    batch_target_flow = torch.cat([batch_target_flow, target_flow], dim=0)

                    if batch_img1.shape[0] == 1:
                        pred_flow = net(batch_img1, batch_img2)
                        # warpped_img2 = warp(batch_img1, pred_flow)
                        pred_flow *= batch_mask1
                        # pred_flow = torch.round(pred_flow)
                        # pred_flow[:, 0] = pred_flow[:, 0] * mask1
                        # pred_flow[:, 1] = pred_flow[:, 1] * mask1
                        # pred_flow = np.round(pred_flow)
                        # pact_mse_loss = patch_mse_loss(batch_target_flow, pred_flow, batch_mask1,
                        #                                patch_size=16, step=16)
                        flow_loss, metrics = flow_loss_fn(batch_img1, batch_img2, batch_target_flow,
                                                          pred_flow, batch_mask1)
                        # pact_mse_loss = patch_mse_loss(batch_img2, warpped_img2, batch_mask1,
                        #                                patch_size=16, step=16)
                        # flow_loss, metrics, = s2lece_loss_criterion(pred_flow, target_flow, train=False)
                        print(flow_loss)
                        print(metrics)
                        compare_flow(target_flow, pred_flow, loss=flow_loss, idx=idx,
                                     path="/home/paxstan/Documents/research_project/code")

                    # pred_last_np = np.floor(pred_flow[-1].detach().squeeze().numpy()).transpose(1, 2, 0).reshape(
                    #     32 * 1024, 2)
                    # invalid_mask = ~valid_mask.flatten()
                    # pred_last_np[invalid_mask, :] = 0
                    #
                    # pred_flow_img = flow_to_color(pred_last_np.reshape(32, 1024, 2))
                    # true_flow_img = flow_to_color(target_flow.detach().squeeze().numpy().transpose(1, 2, 0))
                    #
                    # plt.imshow(flow2rgb(target_flow).transpose(1, 2, 0))
                    # plt.axis('off')
                    # # plt.savefig(f'runs/tg_flow_{idx}.png', format='png', dpi=300, bbox_inches='tight')
                    #
                    # plt.imshow(flow2rgb(pred_last_np).transpose(1, 2, 0))
                    # plt.axis('off')
                    # # plt.savefig(f'runs/pd_flow_{idx}.png', format='png', dpi=300, bbox_inches='tight')
