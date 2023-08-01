from visualization.visualization import compare_flow, visualize_point_cloud, flow_to_color
from matplotlib import pyplot as plt
from input_pipeline.dataloader import normalize_img
import numpy as np
import torch
from models.utils import loss_criterion
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
    with torch.no_grad():
        if type_net == "ae":
            max_count = len(dataloader)
            for idx, input_data in enumerate(dataloader):
                img = input_data.pop('img')
                mask = input_data.pop('mask')
                weight = input_data.pop('weight')
                img = torch.unsqueeze(torch.tensor(img), 0)
                mask = torch.unsqueeze(torch.tensor(mask), 0)
                weight = torch.unsqueeze(torch.tensor(weight), 0)
                # x, skips = net["encoder"](img)
                # pred_img = net["decoder"](x, skips)
                pred_img = net(img)
                recon_loss = F.mse_loss(pred_img, img, reduction='mean')
                print("loss:", recon_loss.mean())

                if idx == max_count - 1:
                    break

        elif type_net == "s2lece":
            for idx, input_data in enumerate(dataloader):
                if idx < 10:
                    img1 = input_data.pop('img1')
                    img2 = input_data.pop('img2')
                    flow = input_data.pop('flow')
                    initial_flow = input_data.pop('initial_flow')
                    mask = input_data.pop('mask')
                    mask1 = input_data.pop('mask1')
                    mask2 = input_data.pop('mask2')
                    img1 = torch.unsqueeze(torch.tensor(img1), 0).unsqueeze(0).to(torch.float32)
                    img2 = torch.unsqueeze(torch.tensor(img2), 0).unsqueeze(0).to(torch.float32)
                    mask = torch.unsqueeze(torch.tensor(mask), 0)
                    target_flow = torch.unsqueeze(torch.tensor(flow), 0)
                    initial_flow = torch.unsqueeze(torch.tensor(initial_flow), 0)
                    # flow_valid_mask = torch.unsqueeze(torch.tensor(flow_valid_mask), 0)
                    # mask1 = torch.unsqueeze(torch.tensor(mask1), 0)
                    # mask2 = torch.unsqueeze(torch.tensor(mask2), 0)
                    pred_flow = net(img1, img2)
                    pred_flow[:, 0] = pred_flow[:, 0] * mask
                    pred_flow[:, 1] = pred_flow[:, 1] * mask
                    flow_loss, metrics, = loss_criterion(initial_flow, pred_flow, target_flow, train=False)
                    print(flow_loss)
                    print(metrics)
                    compare_flow(target_flow, pred_flow, loss=flow_loss)

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

