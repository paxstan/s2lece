import os.path

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models.local_net_ae import ConvolutionAE

iscuda = torch.cuda.is_available()
device = torch.device("cuda" if iscuda else "cpu")


def todevice(x):
    if isinstance(x, dict):
        return {k: todevice(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [todevice(v) for v in x]

    if iscuda:
        return x.contiguous().cuda(non_blocking=True)
    else:
        return x.cpu()


def calculate_nparameters(model):
    def times(shape):
        parameters = 1
        for layer in list(shape):
            parameters *= layer
        return parameters

    layer_params = [times(x.size()) for x in list(model.parameters())]

    return sum(layer_params)


def show_visual_progress(model, test_dataloader, rows=5, flatten=False, vae=False, conditional=False, title=None):
    if title:
        plt.title(title)

    iter(test_dataloader)

    image_org = []
    image_rows = []

    for idx, batch in enumerate(test_dataloader):
        if rows == idx:
            break

        batch = todevice(batch)
        img = batch.pop('img')

        images = model(img).detach().cpu().numpy().reshape(img.size(0), 32, 1024)
        org_im = img.detach().cpu().numpy().reshape(img.size(0), 32, 1024)

        image_idxs = [x for x in range(2)]
        org_images = np.concatenate([images[x].reshape(32, 1024) for x in image_idxs], 1)
        combined_images = np.concatenate([org_im[x].reshape(32, 1024) for x in image_idxs], 1)
        image_org.append(org_images)
        image_rows.append(combined_images)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(np.concatenate(image_org))
    ax2.imshow(np.concatenate(image_rows))
    # plt.imshow(np.concatenate(image_rows))

    if title:
        title = title.replace(" ", "_")
        plt.savefig('runs/' + title)
    # plt.show()


def calculate_loss(model, dataloader, loss_fn=nn.MSELoss()):
    losses = []
    for batch in dataloader:
        batch = todevice(batch)
        img = batch.pop('img')

        loss = loss_fn(img, model(img))

        losses.append(loss)

    return (sum(losses) / len(losses)).item()  # calculate mean


def evaluate(losses, autoencoder, dataloader, flatten=True, vae=False, conditional=False, title=""):
    #     display.clear_output(wait=True)
    if vae and conditional:
        model = lambda x, y: autoencoder(x, y)[0]
    elif vae:
        model = lambda x: autoencoder(x)[0]
    else:
        model = autoencoder

    loss = calculate_loss(model, dataloader)
    print(loss)
    show_visual_progress(model, dataloader, flatten=flatten, vae=vae, conditional=conditional, title=title)
    losses.append(loss)


def train(net, dataloader, test_dataloader, epochs=5, flatten=False, loss_fn=nn.MSELoss(), title=None, config=None):
    optim = torch.optim.Adam(net.parameters())

    train_losses = []
    validation_losses = []
    image_title = ""

    for i in range(epochs):
        for iter, inputs in enumerate(tqdm(dataloader)):
            inputs = todevice(inputs)
            img1 = inputs.pop('img')
            optim.zero_grad()

            output = net(img1)
            loss = loss_fn(img1, output)
            loss.backward()
            optim.step()
            # print(loss.item())

            train_losses.append(loss.item())
        if title:
            image_title = f'{title} - Epoch {i}'
            print(image_title)
        # evaluate(validation_losses, net, test_dataloader, flatten, title=image_title)

    # print(f"\n>> Saving model to {config['save_path']}")
    # torch.save({'net': 'ConvolutionAE()', 'state_dict': net.state_dict()}, config["save_path"])

    print(f"\n>> Saving model to {config['save_path']}")
    encoder_net = net.encoder
    torch.save({'net': 'ConvolutionAE()', 'state_dict': encoder_net.state_dict()}, config["save_path"])
