import logging
from visualization.visualization import compare_flow, visualize_point_cloud, flow_to_color
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
from models.loss import flow_loss_fn, autoencoder_loss_fn
from utils import pytorch_ssim
matplotlib.use('Agg')


def evaluate(net, type_net, input_data, run_paths, random=False):
    """Function to evaluate networks with trained weights"""
    with torch.no_grad():
        if type_net == "ae":
            img = input_data.pop('img')
            mask = input_data.pop('mask')
            img = torch.unsqueeze(torch.tensor(img), 0)
            mask = torch.unsqueeze(torch.tensor(mask), 0)
            pred_img = net(img)
            loss, metrics = autoencoder_loss_fn(pred_img, img, mask)
            logging.info(loss, metrics)
        else:
            path_id = input_data.pop('path')
            img1 = input_data.pop('img1')
            img2 = input_data.pop('img2')
            idx1 = input_data.pop('idx1')
            idx2 = input_data.pop('idx2')
            xyz1 = input_data.pop('xyz1')
            xyz2 = input_data.pop('xyz2')
            initial_flow = input_data.pop('initial_flow')
            target_flow = input_data.pop('flow')
            mask1 = input_data.pop('mask1')
            mask = input_data.pop('mask')
            img1 = torch.unsqueeze(img1, 0).to(torch.float32)
            img2 = torch.unsqueeze(img2, 0).to(torch.float32)
            initial_flow = torch.unsqueeze(torch.tensor(initial_flow), 0)
            target_flow = torch.unsqueeze(torch.tensor(target_flow), 0)
            mask1 = torch.unsqueeze(torch.tensor(mask1), 0)
            pred_flow = net(img1, img2, mask1)
            flow_loss, metrics = flow_loss_fn(img1, img2, target_flow,
                                              pred_flow, mask1, max_flow=600)

            print(flow_loss)
            print(metrics)

            pd_flow = pred_flow[-1]
            magnitude = torch.sum(target_flow ** 2, dim=1).sqrt().unsqueeze(dim=1)
            valid_flow_mask = (magnitude < 600) & (mask1 == 1)
            pd_flow *= valid_flow_mask
            if not random:
                compare_flow(target_flow, pd_flow, idx=path_id, path=run_paths['path_model_id'], loss=flow_loss)
                visualize_point_cloud(pd_flow, initial_flow, idx1, idx2, xyz1, xyz2, mask,
                                      f"{run_paths['path_model_id']}/{path_id}")
            else:
                pd_flow = pred_flow.detach().squeeze().numpy()
                pred_flow_img = flow_to_color(pd_flow.transpose(1, 2, 0))
                plt.imshow(pred_flow_img)
                visualize_point_cloud(pd_flow, initial_flow, idx1, idx2, xyz1, xyz2, mask, run_paths['path_model_id'])


# def random_evaluation(net):
#     """Function to evaluate S2lece network with trained weights with random pairs"""
#     def load(id1, id2):
#         img1 = np.load(f'dataset/data/ae_val_data/{id1}/range.npy')
#         idx1 = np.load(f'dataset/data/ae_val_data/{id1}/idx.npy')
#         xyz1 = np.load(f'dataset/data/ae_val_data/{id1}/xyz.npy')
#         img2 = np.load(f'dataset/data/ae_val_data/{id2}/range.npy')
#         idx2 = np.load(f'dataset/data/ae_val_data/{id2}/idx.npy')
#         xyz2 = np.load(f'dataset/data/ae_val_data/{id2}/xyz.npy')
#         mask2 = np.load(f'dataset/data/ae_val_data/{id2}/valid_mask.npy')
#         return img1, img2, {'idx1': idx1, 'idx2': idx2, 'xyz1': xyz1, 'xyz2': xyz2, 'mask2': mask2}
#
#     img_a, img_b, metadata = load(10, 1000)
#     mask_b = metadata.get('mask2')
#     evaluate(net, img_a, img_b, valid_mask=mask_b, metadata=metadata, random=True)


def test_network(type_net, dataloader, net):
    """Function to test networks before training"""
    l1_lambda = 0.001
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
                    loss, metrics = autoencoder_loss_fn(pred_img, batch_img, batch_mask)

                    l1_regularization = torch.tensor(0.)
                    for param in net.parameters():
                        l1_regularization += torch.norm(param, 1)

                    # Add regularization term to the loss
                    loss += l1_lambda * l1_regularization

                    ssim_loss = metrics["ssim loss"].detach().item()
                    psnr_loss = metrics["psnr loss"].detach().item()
                    diff_loss = metrics["diff loss"].detach().item()
                    mask_loss = metrics["mask loss"].detach().item()

                    print(f"patch loss: {loss}, psnr_loss: {psnr_loss}, ssim: {ssim_loss}, diff_loss:{diff_loss}")
                    batch_img = torch.empty(0)
                    batch_mask = torch.empty(0)

                if idx == max_count - 1:
                    break

        elif type_net == "s2lece":
            nan_list = ['corres_284_280']
            batch_img1 = torch.empty(0)
            batch_img2 = torch.empty(0)
            batch_mask1 = torch.empty(0)
            batch_target_flow = torch.empty(0)
            for idx, input_data in enumerate(dataloader):
                if idx >= 0:
                    path = input_data.pop('path')
                    img1 = input_data.pop('img1')
                    img2 = input_data.pop('img2')
                    target_flow = input_data.pop('flow')
                    mask1 = input_data.pop('mask1')
                    img1 = torch.unsqueeze(img1, 0)
                    img2 = torch.unsqueeze(img2, 0)
                    target_flow = torch.unsqueeze(target_flow, 0)
                    mask1 = torch.unsqueeze(torch.tensor(mask1), 0)
                    if path in nan_list:
                        batch_img1 = torch.cat([batch_img1, img1], dim=0)
                        batch_img2 = torch.cat([batch_img2, img2], dim=0)
                        batch_mask1 = torch.cat([batch_mask1, mask1], dim=0)
                        batch_target_flow = torch.cat([batch_target_flow, target_flow], dim=0)

                    if batch_img1.shape[0] == 1:
                        pred_flow = net(batch_img1, batch_img2, batch_mask1)
                        flow_loss, metrics = flow_loss_fn(batch_img1, batch_img2, batch_target_flow,
                                                          pred_flow, batch_mask1)
                        assert not (torch.isinf(flow_loss) or torch.isnan(
                            flow_loss)), f"Loss value is {flow_loss} at {path}"
                        print(flow_loss)
                        print(metrics)
                        compare_flow(target_flow, pred_flow[-1], loss=flow_loss, idx=idx,
                                     path="/home/paxstan/Documents/research_project/code")
                        batch_img1 = torch.empty(0)
                        batch_img2 = torch.empty(0)
                        batch_mask1 = torch.empty(0)
                        batch_target_flow = torch.empty(0)
                        nan_list.remove(path)
