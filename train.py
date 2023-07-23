import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
from models.utils import loss_criterion
from visualization.visualization import compare_flow
import os
import logging

torch.autograd.set_detect_anomaly(True)


# iscuda = False
# device = torch.device("cuda" if iscuda else "cpu")


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
        plt.savefig('/home/paxstan/Documents/research_project/code/runs/progress/' + title)


def train(train_fe, net, dataloader, test_dataloader, epochs=5, config=None, title=None, is_cuda=False):
    if is_cuda:
        device = torch.device("cuda")
    device = torch.device("cuda" if is_cuda else "cpu")
    # start a new wandb run to track this script


class Train:
    def __init__(self, is_cuda=False):
        self.is_cuda = is_cuda
        self.train_loss = 0
        self.val_loss = 0

    def __call__(self):
        self.train()

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

    @staticmethod
    def save_checkpoint(state, filename):
        torch.save(state, filename)
        return True

    @staticmethod
    def load_checkpoint(checkpoint_dir):
        checkpoint = None
        # Get a list of all files in the directory
        files = os.listdir(checkpoint_dir)

        # Filter out non-checkpoint files if needed
        checkpoint_files = [file for file in files if file.endswith('.pt')]

        # Sort the checkpoint files based on modification time
        sorted_files = sorted(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))

        if len(sorted_files) > 0:
            latest_checkpoint = sorted_files[-1]
            print("Latest checkpoint file:", latest_checkpoint)
            checkpoint = torch.load(latest_checkpoint)
        else:
            print("No checkpoint files found.")
        return checkpoint


class TrainAutoEncoder(Train):
    def __init__(self, net, dataloader, test_dataloader, epochs=5, config=None, title=None, is_cuda=False):
        self.net = net
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.config = config
        self.title = title
        self.learning_rate = 0.05
        self.wandb_config = {
            "learning_rate": self.learning_rate,
            "architecture": "CNN-Autoencoder",
            "dataset": "Hilti exp04",
            "epochs": self.epochs,
        }
        super().__init__(is_cuda=is_cuda)

    def train(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="s2lece",

            # track hyperparameters and run metadata
            config=self.wandb_config
        )
        self.train_autoencoder()
        print(f"\n>> Saving model to {self.config['fe_save_path']}")
        torch.save({'net': 'FeatureExtractorNet()', 'state_dict': self.net.encoder.state_dict()},
                   self.config["fe_save_path"])
        wandb.finish()

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


class TrainSleceNet(Train):
    def __init__(self, net, dataloader, test_dataloader, config, run_paths, is_cuda=False):
        self.net = net
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.epochs = config['epoch']
        self.ckpt_interval = config['ckpt_interval']
        self.ckpt_path = run_paths['path_ckpts_train']
        self.save_path = os.path.join(run_paths['path_model_id'], config["save_path"])
        self.learning_rate = float(config['learning_rate'])
        self.wandb_config = {
            "learning_rate": self.learning_rate,
            "architecture": "SLECE Net",
            "dataset": "Hilti exp04",
            "epochs": self.epochs,
        }
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        checkpoint = self.load_checkpoint(self.ckpt_path)
        if checkpoint is None:
            self.start_epoch = 0
            logging.info("fresh model....")
        else:
            logging.info("checkpoint model...")
            self.start_epoch = checkpoint["epoch"]
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        super().__init__(is_cuda=is_cuda)

    def train(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="s2lece",

            # track hyperparameters and run metadata
            config=self.wandb_config
        )
        for result in self.train_slecenet():
            log = f"epoch: {result['epoch'] + 1}, train loss: {result['train loss']}, val loss: {result['val loss']}"
            print(log)
            logging.info(log)
            wandb.log(result)
            if result['epoch'] % self.ckpt_interval == 0:
                logging.info(f"Saving checkpoint at epoch {result['epoch']+1}.")
                torch.save({
                    'epoch': result['epoch']+1,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.train_loss,
                }, os.path.join(self.ckpt_path, f"ckpts_{result['epoch']+1}.pt"))
        torch.save({'net': 'SleceNet', 'state_dict': self.net.state_dict()}, self.save_path)
        log = f"Saved model to {self.save_path}"
        print(log)
        logging.info(log)
        wandb.finish()

    def train_slecenet(self):
        train_losses = []
        for i in range(self.start_epoch, self.epochs):
            for _, inputs in enumerate(tqdm(self.dataloader)):
                inputs = self.todevice(inputs)
                img1 = inputs.pop('img1')
                img2 = inputs.pop('img2')
                target_flow = inputs.pop('aflow')
                mask2 = inputs.pop('mask2')
                pred_flow = self.net(img1, img2)
                flow_loss, metrics, _ = loss_criterion(pred_flow, target_flow, mask2, img1, img2)

                self.optimizer.zero_grad()
                flow_loss.backward()
                self.optimizer.step()

                loss = flow_loss.detach().item()
                train_losses.append(loss)
                del flow_loss
                torch.cuda.empty_cache()
            self.train_loss = (sum(train_losses) / len(train_losses))
            self.evaluate_slece(i_epoch=i)
            yield {"train loss": self.train_loss, "val loss": self.val_loss, "epoch": i}

    def evaluate_slece(self, i_epoch):
        losses = []
        for idx, inputs in enumerate(self.test_dataloader):
            with torch.no_grad():
                inputs = self.todevice(inputs)
                img1 = inputs.pop('img1')
                img2 = inputs.pop('img2')
                target_flow = inputs.pop('aflow')
                mask2 = inputs.pop('mask2')
                pred_flow = self.net(img1, img2)
                flow_loss, metrics, valid_masks = loss_criterion(pred_flow, target_flow, mask2, img1, img2)
                if idx == 1:
                    compare_flow(target_flow, pred_flow, valid_masks, loss=flow_loss, idx=i_epoch)
                losses.append(flow_loss.detach().item())

        self.val_loss = (sum(losses) / len(losses))  # calculate mean
