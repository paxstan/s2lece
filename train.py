import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch.optim.lr_scheduler as toptim
from models.model_utils import s2lece_loss_criterion, ae_loss_criterion, CustomMSELoss
from visualization.visualization import compare_flow, show_visual_progress
import os
import logging
from torch.cuda.amp import GradScaler

torch.autograd.set_detect_anomaly(True)


class Train:
    def __init__(self, net, config, run_paths, model_type, is_cuda=False):
        self.model_type = model_type
        self.net = net
        self.ckpt_interval = config[model_type]['ckpt_interval']
        self.ckpt_path = run_paths['path_ckpts_train']
        self.progress_path = run_paths['path_model_progress']
        self.save_path = os.path.join(run_paths['path_model_id'], config[model_type]["save_path"])
        self.epochs = config[model_type]['epoch']
        self.learning_rate = float(config[model_type]['learning_rate'])
        self.early_stop_limit = config[model_type]['early_stopping']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.is_cuda = is_cuda
        self.train_loss = 0
        self.val_loss = 0
        self.best_loss = 0
        self.stop_counter = 0
        self.wandb_config = {
            "learning_rate": self.learning_rate,
            "architecture": model_type,
            "dataset": "Hilti exp04",
            "epochs": self.epochs,
            "run_id": run_paths['path_model_id']
        }
        self.enable_wandb = config[model_type]['enable_wandb']

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
            # print(log)
            logging.info(log)
            if self.enable_wandb:
                wandb.log(result)
            if result['val loss'] < self.best_loss or result["epoch"] == 0:
                self.best_loss = result['val loss']
                logging.info(f"best loss: {self.best_loss}")
                torch.save({
                    'epoch': result['epoch'] + 1,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.train_loss,
                    'val loss': self.val_loss
                }, os.path.join(self.ckpt_path, "best_model.pt"))
                self.stop_counter = 0
            else:
                self.stop_counter += 1
            # if self.early_stop_limit <= self.stop_counter:
            #     print("Stopping early!!!!!")
            #     break

            if result['epoch'] % self.ckpt_interval == 0:
                logging.info(f"Saving checkpoint at epoch {result['epoch'] + 1}.")
                torch.save({
                    'epoch': result['epoch'] + 1,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.train_loss,
                    'val loss': self.val_loss
                }, os.path.join(self.ckpt_path, f"ckpts_{result['epoch'] + 1}.pt"))

        if self.model_type == "autoencoder":
            torch.save({'net': 'FeatureExtractorNet', 'state_dict': self.net.state_dict()}, self.save_path)
        else:
            torch.save({'net': 'SleceNet', 'state_dict': self.net.state_dict()}, self.save_path)

        log = f"Saved model to {self.save_path}"
        # print(log)
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
            logging.info(f"Latest checkpoint file:{latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
        else:
            logging.info("No checkpoint files found.")
        return checkpoint


class TrainAutoEncoder(Train):
    def __init__(self, net, train_loader, val_loader, run_paths, config=None, title=None, is_cuda=False,
                 max_count=0):
        super().__init__(net=net, config=config, run_paths=run_paths, is_cuda=is_cuda, model_type="autoencoder")
        self.dataloader = train_loader
        self.val_loader = val_loader
        self.loss_func = CustomMSELoss()
        self.train_dicts = []
        self.train_dicts.append({'params': self.net.encoder.parameters()})
        self.train_dicts.append({'params': self.net.decoder.parameters()})
        self.train_dicts.append({'params': self.net.head.parameters()})
        self.optimizer = torch.optim.SGD(self.train_dicts,
                                         lr=config["autoencoder"]["learning_rate"],
                                         momentum=config["autoencoder"]["momentum"],
                                         weight_decay=config["autoencoder"]["weight_decay"])
        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = len(self.dataloader)
        up_steps = int(config["autoencoder"]["wup_epochs"] * steps_per_epoch)
        final_decay = config["autoencoder"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                 lr=config["autoencoder"]["learning_rate"],
                                 warmup_steps=up_steps,
                                 momentum=config["autoencoder"]["momentum"],
                                 decay=final_decay)
        checkpoint = self.load_checkpoint(self.ckpt_path)
        self.start_epoch = 0
        if checkpoint is None:
            logging.info("fresh model....")
        else:
            logging.info("checkpoint model...")
            if checkpoint["epoch"] < self.epochs:
                self.start_epoch = checkpoint["epoch"]
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.title = title
        self.max_count = max_count

    @staticmethod
    def loss_fn(recon_x, x, mask):
        # Reconstruction loss
        recon_loss = ae_loss_criterion(recon_x, x, mask)
        # recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        return recon_loss

    def train_epoch(self):
        image_title = ""
        for i in range(self.start_epoch, self.epochs):
            self.net.train()
            train_losses = 0.0
            for idx, inputs in enumerate(tqdm(self.dataloader)):
                path = inputs.pop('path')
                inputs = self.todevice(inputs)
                img = inputs.pop('img')
                mask = inputs.pop('mask')
                pred = self.net(img)
                loss = self.loss_func(pred, img, mask)
                if torch.isinf(loss) or torch.isnan(loss):
                    print(loss)
                    print(path)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses += loss.detach().item()
                # del loss, pred
                # torch.cuda.empty_cache()
            if self.title:
                image_title = f'{self.title} - Epoch {i}'
            before_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            after_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Epoch %d: SGD lr %.4f -> %.4f" % (i, before_lr, after_lr))
            self.train_loss = train_losses / len(self.dataloader)
            self.evaluate_autoencoder(title=image_title, epoch=i)
            yield {"train (running) loss": self.train_loss, "val loss": self.val_loss, "epoch": i}

    def evaluate_autoencoder(self, title, epoch, progress_view=True):
        logging.info(f"evaluation at epoch: {epoch}")
        losses = 0.0
        val_losses = 0.0
        self.net.eval()
        for idx, batch in enumerate(tqdm(self.dataloader)):
            path = batch.pop('path')
            batch = self.todevice(batch)
            img = batch.pop('img')
            mask = batch.pop('mask')
            pred = self.net(img)
            loss = self.loss_func(pred, img, mask)
            losses += loss.detach().item()
            if torch.isinf(loss) or torch.isnan(loss):
                print(loss)
                print(path)
        for idx, batch in enumerate(tqdm(self.val_loader)):
            path = batch.pop('path')
            batch = self.todevice(batch)
            img = batch.pop('img')
            mask = batch.pop('mask')
            pred = self.net(img)
            loss = self.loss_func(pred, img, mask)
            val_losses += loss.detach().item()
            if torch.isinf(loss) or torch.isnan(loss):
                print(loss)
                print(path)
            if progress_view:
                show_visual_progress(img, pred, self.progress_path, title=title, loss=loss)
                progress_view = False
        # with torch.no_grad():

        # del loss, pred
        # torch.cuda.empty_cache()
        print(f"validation training dataset: {losses / len(self.dataloader)}")
        self.val_loss = val_losses / len(self.val_loader)


class TrainSleceNet(Train):
    def __init__(self, net, dataloader, test_dataloader, config, run_paths, device, is_cuda=False):
        super().__init__(net=net, config=config, run_paths=run_paths, is_cuda=is_cuda, model_type="s2lece")
        self.device = device
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.mixed_precision = config["s2lece"]["mixed_precision"]
        steps_per_epoch = len(self.dataloader)
        self.train_dicts = []
        self.train_dicts.append({'params': self.net.encoder.parameters()})
        self.train_dicts.append({'params': self.net.context_net.parameters()})
        self.train_dicts.append({'params': self.net.update.parameters()})
        # self.optimizer = torch.optim.AdamW(self.train_dicts,
        #                                   lr=self.learning_rate, weight_decay=float(config["s2lece"]["weight_decay"]),
        #                                    eps=float(config["s2lece"]["epsilon"]))
        self.optimizer = torch.optim.SGD(self.train_dicts,
                                         lr=config["s2lece"]["learning_rate"],
                                         momentum=config["s2lece"]["momentum"],
                                         weight_decay=config["s2lece"]["weight_decay"])
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
        #                                                      self.learning_rate, steps_per_epoch,
        #                                                      pct_start=0.05, cycle_momentum=False,
        #                                                      anneal_strategy='linear')
        steps_per_epoch = len(self.dataloader)
        up_steps = int(config["s2lece"]["wup_epochs"] * steps_per_epoch)
        final_decay = config["s2lece"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=config["s2lece"]["learning_rate"],
                                  warmup_steps=up_steps,
                                  momentum=config["s2lece"]["momentum"],
                                  decay=final_decay)
        # self.scaler = GradScaler(enabled=self.mixed_precision)
        self.clip = config["s2lece"]["clip"]
        checkpoint = self.load_checkpoint(self.ckpt_path)
        if checkpoint is None:
            self.start_epoch = 0
            logging.info("fresh model....")
        else:
            logging.info("checkpoint model...")
            self.start_epoch = checkpoint["epoch"]
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train_epoch(self):
        train_losses = []
        for i in range(self.start_epoch, self.epochs):
            self.net.train()
            for _, inputs in enumerate(tqdm(self.dataloader)):
                inputs = self.todevice(inputs)
                img1 = inputs.pop('img1')
                img2 = inputs.pop('img2')
                target_flow = inputs.pop('flow')
                mask1 = inputs.pop('mask1')
                pred_flow = self.net(img1, img2)
                # pred_flow *= mask1
                flow_loss, metrics = s2lece_loss_criterion(pred_flow, target_flow)
                # logging.info(metrics)
                # with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
                #     pred_flow = self.net(img1, img2)
                #     pred_flow *= mask1
                #     flow_loss, metrics = s2lece_loss_criterion(pred_flow, target_flow)

                self.optimizer.zero_grad()
                flow_loss.backward()
                # self.scaler.scale(flow_loss).backward()
                # self.scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                # self.scaler.step(self.optimizer)
                self.optimizer.step()
                # self.scheduler.step()
                # self.scaler.update()

                loss = flow_loss.detach().item()
                train_losses.append(loss)
                del flow_loss
                torch.cuda.empty_cache()

            # print(f"train has nan: {has_nan(train_losses)}")
            # print(f"train has sum: {train_losses}")
            before_lr = self.optimizer.param_groups[0]["lr"]
            after_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            logging.info("Epoch %d: SGD lr %.6f -> %.6f" % (i, before_lr, after_lr))
            self.train_loss = (sum(train_losses) / len(train_losses))
            self.evaluate_slece(i_epoch=i)
            yield {"train loss": self.train_loss, "val loss": self.val_loss, "epoch": i}

    def evaluate_slece(self, i_epoch, progress_view=True):
        logging.info(f"evaluation at epoch: {i_epoch}")
        losses = []
        self.net.eval()
        with torch.no_grad():
            for idx, inputs in enumerate(tqdm(self.test_dataloader)):
                inputs = self.todevice(inputs)
                img1 = inputs.pop('img1')
                img2 = inputs.pop('img2')
                target_flow = inputs.pop('flow')
                mask1 = inputs.pop('mask1')
                pred_flow = self.net(img1, img2)
                pred_flow *= mask1
                # pred_flow[:, 0] = pred_flow[:, 0] * mask[:, 0]
                # pred_flow[:, 1] = pred_flow[:, 1] * mask[:, 0]
                flow_loss, metrics = s2lece_loss_criterion(pred_flow, target_flow, train=False)
                if progress_view:
                    compare_flow(target_flow, pred_flow, self.progress_path, loss=flow_loss, idx=i_epoch)
                    progress_view = False
                losses.append(flow_loss.detach().item())

        # print(f"val has nan: {has_nan(losses)}")
        # print(f"val has sum: {sum(losses)}")
        self.val_loss = (sum(losses) / len(losses))  # calculate mean


class warmupLR(toptim.LRScheduler):
    """ Warmup learning rate scheduler.
      Initially, increases the learning rate from 0 to the final value, in a
      certain number of steps. After this number of steps, each step decreases
      LR exponentially.
  """

    def __init__(self, optimizer, lr, warmup_steps, momentum, decay):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum
        self.decay = decay

        # cap to one
        if self.warmup_steps < 1:
            self.warmup_steps = 1

        # cyclic lr
        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        # our params
        self.last_epoch = -1  # fix for pytorch 1.1 and below
        self.finished = False  # am i done
        super().__init__(optimizer)

    def get_lr(self):
        return [self.lr * (self.decay ** self.last_epoch) for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for lr in self.base_lrs]
                self.finished = True
            return super(warmupLR, self).step(epoch)
        else:
            return self.initial_scheduler.step(epoch)


def has_nan(lst):
    for item in lst:
        if isinstance(item, torch.Tensor) and torch.isnan(item).any():
            return True
    return False
