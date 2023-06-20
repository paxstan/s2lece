from visualization.visualization import compare_flow, visualize_point_cloud
from matplotlib import pyplot as plt
from input_pipeline.dataloader import normalize_img
import numpy as np
import torch
from models.utils import loss_criterion
import torch.nn.functional as F


def evaluate(net, img1, img2, flow=None, valid_mask=None, idx=None, metadata=None, random=False):
    img1 = normalize_img(np.array(img1))
    img2 = normalize_img(np.array(img2))
    img1 = torch.unsqueeze(torch.tensor(img1), 0)
    img2 = torch.unsqueeze(torch.tensor(img2), 0)
    pred_flow = net(img1, img2)
    if not random:
        target_flow = torch.unsqueeze(torch.tensor(flow), 0)
        valid_mask_t = torch.unsqueeze(torch.tensor(valid_mask), 0)
        flow_loss, metrics = loss_criterion(pred_flow, target_flow, valid_mask_t, train=False)
        print(flow_loss)
        print(metrics)
        compare_flow(target_flow, pred_flow[-1], valid_mask, idx, flow_loss)
    # else:
    #     pd_flow = pred_flow[-1].detach().squeeze().numpy().transpose(1, 2, 0)
    #     plt.show(pd_flow)
    visualize_point_cloud(pred_flow[-1], metadata)


def test_network(type_net, dataloader, net):
    with torch.no_grad():
        if type_net == "ae":
            for idx, input_data in enumerate(dataloader):
                if idx > 1:
                    img = input_data.pop('img')
                    img = torch.unsqueeze(torch.tensor(img), 0)
                    pred_img, mu, logvar = net(img)
                    recon_loss = F.mse_loss(pred_img, img, reduction='mean')
                    print("loss:", recon_loss.mean())
        elif type_net == "s2lece":
            for idx, input_data in enumerate(dataloader):
                if idx < 10:
                    img1 = input_data.pop('img1')
                    img2 = input_data.pop('img2')
                    flow = input_data.pop('aflow')
                    flow_valid_mask = input_data.pop('flow_mask')
                    mask1 = input_data.pop('mask1')
                    mask2 = input_data.pop('mask2')
                    img1 = torch.unsqueeze(torch.tensor(img1), 0)
                    img2 = torch.unsqueeze(torch.tensor(img2), 0)
                    target_flow = torch.unsqueeze(torch.tensor(flow), 0)
                    flow_valid_mask = torch.unsqueeze(torch.tensor(flow_valid_mask), 0)
                    mask1 = torch.unsqueeze(torch.tensor(mask1), 0)
                    mask2 = torch.unsqueeze(torch.tensor(mask2), 0)
                    pred_flow = net(img1, img2)
                    flow_loss, metrics = loss_criterion(pred_flow, target_flow, flow_valid_mask)
                    print(flow_loss)
                    print(metrics)
                    compare_flow(target_flow, pred_flow[-1], flow_valid_mask)

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

