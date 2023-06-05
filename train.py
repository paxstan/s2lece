import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F


# iscuda = False
# device = torch.device("cuda" if iscuda else "cpu")


# def calculate_nparameters(model):
#     def times(shape):
#         parameters = 1
#         for layer in list(shape):
#             parameters *= layer
#         return parameters
#
#     layer_params = [times(x.size()) for x in list(model.parameters())]
#
#     return sum(layer_params)


def show_visual_progress(org_img, pred_img, title=None):
    if title:
        plt.title(title)

    org_img = org_img.detach().cpu().numpy()[0, 0, :, :].reshape(32, 1024)
    pred_img = pred_img.detach().cpu().numpy()[0, 0, :, :].reshape(32, 1024)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(org_img)
    ax2.imshow(pred_img)

    if title:
        title = title.replace(" ", "_")
        plt.savefig('runs/progress/' + title)


def train(train_fe, net, dataloader, test_dataloader, epochs=5, config=None, title=None, is_cuda=False):
    if is_cuda:
        device = torch.device("cuda")
    device = torch.device("cuda" if is_cuda else "cpu")
    # start a new wandb run to track this script


class Train:
    def __init__(self, wandb_config, is_cuda=False):
        self.is_cuda = is_cuda
        self.wandb_config = wandb_config
        self.train_loss = 0
        self.val_loss = 0

    def __call__(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="s2lece",

            # track hyperparameters and run metadata
            config=self.wandb_config
        )

        self.train()
        wandb.finish()

    def train(self):
        raise NotImplementedError("train() must be implemented in the child class")

    def todevice(self, x):
        if isinstance(x, dict):
            return {k: self.todevice(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [self.todevice(v) for v in x]

        if self.is_cuda:
            return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()


class TrainAutoEncoder(Train):
    def __init__(self, net, dataloader, test_dataloader, epochs=5, config=None, title=None, is_cuda=False):
        self.net = net
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.config = config
        self.title = title
        self.learning_rate = 0.05
        wandb_config = {
            "learning_rate": self.learning_rate,
            "architecture": "CNN-Autoencoder",
            "dataset": "Hilti exp04",
            "epochs": self.epochs,
        }
        super().__init__(is_cuda=is_cuda, wandb_config=wandb_config)

    def train(self):
        self.train_autoencoder()
        print(f"\n>> Saving model to {self.config['fe_save_path']}")
        torch.save({'net': 'FeatureExtractorNet()', 'state_dict': self.net.encoder.state_dict()},
                   self.config["fe_save_path"])

    def loss_fn(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        if self.net.vae:
            # Kullback-Leibler Divergence loss
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss
            recon_loss = recon_loss + kld_loss
        return recon_loss

    def train_autoencoder(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        train_losses = []
        image_title = ""

        for i in range(self.epochs):
            print(f"epoch: {i}")
            for iter, inputs in enumerate(tqdm(self.dataloader)):
                inputs = self.todevice(inputs)
                img = inputs.pop('img')

                optimizer.zero_grad()
                pred, mu, logvar = self.net(img)
                loss = self.loss_fn(pred, img, mu, logvar)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().item())
                del loss, pred
                torch.cuda.empty_cache()
            if self.title:
                image_title = f'{self.title} - Epoch {i}'
            self.train_loss = (sum(train_losses) / len(train_losses))
            self.evaluate_autoencoder(title=image_title)
            print(f"train loss: {self.train_loss}, val loss: {self.val_loss}")
            wandb.log({"train loss": self.train_loss, "val loss": self.val_loss})

    def evaluate_autoencoder(self, title, progress_view=True):
        losses = []
        for idx, batch in enumerate(self.test_dataloader):
            batch = self.todevice(batch)
            img = batch.pop('img')
            with torch.no_grad():
                pred, mu, logvar = self.net(img)
                if progress_view:
                    show_visual_progress(img, pred, title)
                    progress_view = False
                loss = self.loss_fn(pred, img, mu, logvar)
                losses.append(loss.detach().item())
                del loss, pred
                torch.cuda.empty_cache()
        self.val_loss = sum(losses) / len(losses)


class TrainFlowModel(Train):
    def __init__(self, net, dataloader, test_dataloader, epochs=5, config=None, is_cuda=False):
        self.net = net
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.config = config
        wandb_config = {
            "learning_rate": 1e-4,
            "architecture": "FlowModel",
            "dataset": "Hilti exp04",
            "epochs": self.epochs,
        }
        self.param_groups = [{'params': self.net.bias_parameters(), 'weight_decay': 0},
                             {'params': self.net.weight_parameters(), 'weight_decay': 4e-4}]
        # self.optimizer = torch.optim.Adam(self.param_groups, 0.9,
        #                                   betas=(0.9, 0.999))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150, 200], gamma=0.5)
        super().__init__(is_cuda=is_cuda, wandb_config=wandb_config)

    def train(self):
        self.train_flownet()
        print(f"\n>> Saving model to {self.config['save_path']}")
        torch.save({'net': 'FlowModel()', 'state_dict': self.net.state_dict()}, self.config["save_path"])

    @staticmethod
    def loss_fn(pred_flow, gt_flow):
        loss = torch.norm(pred_flow - gt_flow, p=2, dim=1)
        flow_mask = (gt_flow[:, 0] == 0) & (gt_flow[:, 1] == 0)
        loss = loss[~flow_mask].mean()
        return loss

    def train_flownet(self):
        train_losses = []

        for i in range(self.epochs):
            for _, inputs in enumerate(tqdm(self.dataloader)):
                inputs = self.todevice(inputs)
                img1 = inputs.pop('img1')
                img2 = inputs.pop('img2')
                target_flow = inputs.pop('aflow')

                pred_flow = self.net(img1, img2)
                loss = self.loss_fn(pred_flow, target_flow)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.detach().item())
                del loss
                torch.cuda.empty_cache()
            self.train_loss = (sum(train_losses) / len(train_losses))
            self.evaluate_flow_model()
            print(f"train loss: {self.train_loss}, val loss: {self.val_loss}")
            wandb.log({"train loss": self.train_loss, "val loss": self.val_loss})
            # scheduler.step()

    def evaluate_flow_model(self):
        losses = []
        for idx, inputs in enumerate(self.test_dataloader):
            inputs = self.todevice(inputs)
            img1 = inputs.pop('img1')
            img2 = inputs.pop('img2')
            target_flow = inputs.pop('aflow')

            with torch.no_grad():
                pred_flow = self.net(img1, img2)
                loss = self.loss_fn(pred_flow, target_flow)
                losses.append(loss.detach().item())

        self.val_loss = (sum(losses) / len(losses))  # calculate mean
