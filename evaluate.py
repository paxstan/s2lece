from visualization.visualization import flow_to_color
from matplotlib import pyplot as plt
from input_pipeline.dataloader import normalize_img
import numpy as np
import torch


def evaluate(net, img1, img2, flow):
    img1 = normalize_img(np.array(img1))
    img2 = normalize_img(np.array(img2))
    img1 = torch.unsqueeze(torch.tensor(img1), 0)
    img2 = torch.unsqueeze(torch.tensor(img2), 0)
    pred_flow = net(img1, img2)
    pred_flow_img = flow_to_color(pred_flow.detach().squeeze().numpy().transpose(1, 2, 0))
    true_flow_img = flow_to_color(flow.transpose(1, 2, 0))

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    axes[0].imshow(true_flow_img.squeeze())
    axes[0].set_title("original flow")

    axes[1].imshow(pred_flow_img.squeeze())
    axes[1].set_title("predicted flow")

    plt.tight_layout()
    plt.show()
