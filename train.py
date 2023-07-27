import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from models.utils import loss_criterion
from visualization.visualization import compare_flow, show_visual_progress
import os
import logging

torch.autograd.set_detect_anomaly(True)


class Train:
    def __init__(self, net, config, run_paths, model_type, is_cuda=False, enable_wandb=False):
        self.model_type = model_type
        self.net = net
        self.ckpt_interval = config[model_type]['ckpt_interval']
        self.ckpt_path = run_paths['path_ckpts_train']
        self.progress_path = run_paths['path_model_progress']
        self.save_path = os.path.join(run_paths['path_model_id'], config[model_type]["save_path"])
        self.epochs = config[model_type]['epoch']
        self.learning_rate = float(config[model_type]['learning_rate'])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.is_cuda = is_cuda
        self.train_loss = 0
        self.val_loss = 0
        self.wandb_config = {
            "learning_rate": self.learning_rate,
            "architecture": model_type,
            "dataset": "Hilti exp04",
            "epochs": self.epochs,
        }
        self.enable_wandb = enable_wandb

    def __call__(self):
        self.train()

    def train(self):
        if self.enable_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="s2lece",

                # track hyperparameters and run metadata
                config=self.wandb_config
            )
        for result in self.train_epoch():
            log = f"epoch: {result['epoch'] + 1}, train loss: {result['train loss']}, val loss: {result['val loss']}"
            print(log)
            logging.info(log)
            if self.enable_wandb:
                wandb.log(result)
            if result['epoch'] % self.ckpt_interval == 0:
                logging.info(f"Saving checkpoint at epoch {result['epoch'] + 1}.")
                torch.save({
                    'epoch': result['epoch'] + 1,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.train_loss,
                }, os.path.join(self.ckpt_path, f"ckpts_{result['epoch'] + 1}.pt"))

        if self.model_type == "autoencoder":
            torch.save({'net': 'FeatureExtractorNet()', 'state_dict': self.net.encoder.state_dict()}, self.save_path)
        else:
            torch.save({'net': 'SleceNet', 'state_dict': self.net.state_dict()}, self.save_path)

        log = f"Saved model to {self.save_path}"
        print(log)
        logging.info(log)
        if self.enable_wandb:
            wandb.finish()

    def train_epoch(self):
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
    def __init__(self, net, dataloader, test_dataloader, run_paths, config=None, title=None, is_cuda=False,
                 max_count=0):
        super().__init__(net=net, config=config, run_paths=run_paths, is_cuda=is_cuda, model_type="autoencoder")
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        checkpoint = self.load_checkpoint(self.ckpt_path)
        if checkpoint is None:
            self.start_epoch = 0
            logging.info("fresh model....")
        else:
            logging.info("checkpoint model...")
            self.start_epoch = checkpoint["epoch"]
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.title = title
        self.max_count = max_count

    @staticmethod
    def loss_fn(recon_x, x):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        return recon_loss

    def train_epoch(self):
        train_losses = []
        image_title = ""

        for i in range(self.start_epoch, self.epochs):
            print(f"epoch: {i}")
            for idx, inputs in enumerate(tqdm(self.dataloader)):
                inputs = self.todevice(inputs)
                img = inputs.pop('img')
                mask = inputs.pop('mask')
                weight = inputs.pop('weight')
                self.optimizer.zero_grad()
                pred = self.net(img)
                loss = self.loss_fn(pred, img)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.detach().item())
                del loss, pred
                torch.cuda.empty_cache()
                if idx == self.max_count - 1:
                    break
            if self.title:
                image_title = f'{self.title} - Epoch {i}'
            self.train_loss = (sum(train_losses) / len(train_losses))
            self.evaluate_autoencoder(title=image_title)
            yield {"train loss": self.train_loss, "val loss": self.val_loss, "epoch": i}

    def evaluate_autoencoder(self, title, progress_view=True):
        losses = []
        for idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                batch = self.todevice(batch)
                img = batch.pop('img')
                pred = self.net(img)
                loss = self.loss_fn(pred, img)
                losses.append(loss.detach().item())
                if progress_view:
                    show_visual_progress(img, pred, self.progress_path, title=title, loss=loss)
                    progress_view = False
                del loss, pred
                torch.cuda.empty_cache()
        self.val_loss = sum(losses) / len(losses)


class TrainSleceNet(Train):
    def __init__(self, net, dataloader, test_dataloader, config, run_paths, is_cuda=False):
        super().__init__(net=net, config=config, run_paths=run_paths, is_cuda=is_cuda, model_type="s2lece")
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
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

    # def train(self):
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="s2lece",
    #
    #         # track hyperparameters and run metadata
    #         config=self.wandb_config
    #     )
    #     for result in self.train_epoch():
    #         log = f"epoch: {result['epoch'] + 1}, train loss: {result['train loss']}, val loss: {result['val loss']}"
    #         print(log)
    #         logging.info(log)
    #         wandb.log(result)
    #         if result['epoch'] % self.ckpt_interval == 0:
    #             logging.info(f"Saving checkpoint at epoch {result['epoch'] + 1}.")
    #             torch.save({
    #                 'epoch': result['epoch'] + 1,
    #                 'model_state_dict': self.net.state_dict(),
    #                 'optimizer_state_dict': self.optimizer.state_dict(),
    #                 'loss': self.train_loss,
    #             }, os.path.join(self.ckpt_path, f"ckpts_{result['epoch'] + 1}.pt"))
    #     torch.save({'net': 'SleceNet', 'state_dict': self.net.state_dict()}, self.save_path)
    #     log = f"Saved model to {self.save_path}"
    #     print(log)
    #     logging.info(log)
    #     wandb.finish()

    def train_epoch(self):
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
