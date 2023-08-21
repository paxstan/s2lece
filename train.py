import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch.optim.lr_scheduler as toptim
from models.model_utils import s2lece_loss_criterion, ae_loss_criterion, CustomMSELoss, patch_mse_loss
from utils import pytorch_ssim
from visualization.visualization import compare_flow, show_visual_progress
import os
import logging
from models.loss import flow_loss_fn
from torch.cuda.amp import GradScaler

torch.autograd.set_detect_anomaly(True)


class Train:
    def __init__(self, net, config, run_paths, model_type, is_cuda=False):
        self.is_cuda = is_cuda
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
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=0.8)

        self.start_epoch = 0
        self.best_loss = 0
        self.stop_counter = 0
        self.wandb_config = {
            "learning_rate": self.learning_rate,
            "architecture": model_type,
            "dataset": config["dataset"][config["datasets"][config["dataset_choice"]]],
            "epochs": self.epochs,
            "run_id": run_paths['path_model_id']
        }
        self.enable_wandb = config[model_type]['enable_wandb']

    def __call__(self):
        self.train()

    def train(self):
        try:
            if self.enable_wandb:
                wandb.init(
                    project="s2lece",
                    config=self.wandb_config
                )

            for i in range(self.start_epoch, self.epochs):
                train_result = self.train_step(i)
                val_result = self.validate_step(i)
                result = {**train_result, **val_result}
                log = (f"epoch: {i + 1}, train loss: {result['training loss']}, "
                       f"val loss: {result['validation loss']}")

                logging.info(result)
                if self.enable_wandb:
                    wandb.log(result)

                if result['validation loss'] < self.best_loss or i == 0:
                    self.best_loss = result['validation loss']
                    logging.info(f"best loss: {self.best_loss}")
                    torch.save({
                        'epoch': i,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': result['training loss'],
                        'val loss': result['validation loss']
                    }, os.path.join(self.ckpt_path, "best_model.pt"))
                    self.stop_counter = 0
                else:
                    self.stop_counter += 1
                # if self.early_stop_limit <= self.stop_counter:
                #     print("Stopping early!!!!!")
                #     break

                if i % self.ckpt_interval == 0:
                    logging.info(f"Saving checkpoint at epoch {i}.")
                    torch.save({
                        'epoch': i,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': result['training loss'],
                        'val loss': result['validation loss']
                    }, os.path.join(self.ckpt_path, f"ckpts_{i}.pt"))

            if self.model_type == "autoencoder":
                torch.save({'net': 'FeatureExtractorNet', 'state_dict': self.net.state_dict()}, self.save_path)
            else:
                torch.save({'net': 'SleceNet', 'state_dict': self.net.state_dict()}, self.save_path)

            log = f"Saved model to {self.save_path}"
            logging.info(log)
            if self.enable_wandb:
                wandb.finish()
        except Exception as e:
            logging.error(f"Exception at main train: {e}")

    def train_step(self, epoch):
        raise NotImplementedError("train_step() must be implemented in the child class")

    def validate_step(self, epoch, progress_view=True):
        raise NotImplementedError("validate_step() must be implemented in the child class")

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
        self.optimizer = torch.optim.SGD(self.net.parameters(),
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

        # self.optimizer = torch.optim.Adam(self.net.parameters(),
        #                                   lr=config["autoencoder"]["learning_rate"],
        #                                   betas=(0.9, 0.999),
        #                                   weight_decay=config["autoencoder"]["weight_decay"],
        #                                   eps=1e-8)
        self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                           lr=self.learning_rate,
                                           weight_decay=float(config["autoencoder"]["weight_decay"]),
                                           eps=1e-8)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer,
        #                                                    base_lr=config["autoencoder"]["learning_rate"],
        #                                                    max_lr=0.0068, step_size_up=30,
        #                                                    step_size_down=60, cycle_momentum=False)

        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
        #                                                      max_lr=0.0068,
        #                                                      epochs=self.epochs,
        #                                                      steps_per_epoch=steps_per_epoch,
        #                                                      cycle_momentum=False,
        #                                                      anneal_strategy='linear')

        self.ssim_loss = pytorch_ssim.SSIM(window_size=16)

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
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.title = title
        self.max_count = max_count

    def loss_fn(self, pred, img, mask):
        # Reconstruction loss
        patch_loss = patch_mse_loss(pred, img, mask)
        mse_loss = ae_loss_criterion(pred, img, mask)
        ssim_metric = self.ssim_loss(img, pred)
        return {"loss": patch_loss, "mse": mse_loss, "ssim": ssim_metric}

    def train_step(self, epoch):
        image_title = ""
        for i in range(self.start_epoch, self.epochs):
            self.net.train()
            train_losses = 0.0
            mse_loss = 0.0
            ssim = 0.0
            after_lr = 0.0
            try:
                for idx, inputs in enumerate(tqdm(self.dataloader)):
                    path = inputs.pop('path')
                    inputs = self.todevice(inputs)
                    img = inputs.pop('img')
                    mask = inputs.pop('mask')

                    self.optimizer.zero_grad()
                    pred = self.net(img)
                    losses = self.loss_fn(pred, img, mask)
                    loss = losses["loss"]
                    if torch.isinf(loss) or torch.isnan(loss):
                        print(loss)
                        print(path)
                        break
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    train_losses += loss.detach().item()
                    mse_loss += losses["mse"].detach().item()
                    ssim += losses["ssim"].detach().item()
                    # del loss, pred
                    # torch.cuda.empty_cache()
                if self.title:
                    image_title = f'{self.title} - Epoch {i}'
                before_lr = self.optimizer.param_groups[0]["lr"]
                after_lr = self.optimizer.param_groups[0]["lr"]
                logging.info("Epoch %d: Optimizer lr %.4f -> %.4f" % (i, before_lr, after_lr))
                self.train_loss = train_losses / len(self.dataloader)
                self.evaluate_autoencoder(title=image_title, epoch=i)
            except Exception as e:
                logging.error(f"Exception at train: {e}")
            yield {"epoch": i,
                   "training loss": self.train_loss,
                   "validation loss": self.val_loss,
                   "ssim": ssim / len(self.dataloader),
                   "mse": mse_loss / len(self.dataloader),
                   "learning rate": after_lr}

    def evaluate_autoencoder(self, title, epoch, progress_view=True):
        logging.info(f"evaluation at epoch: {epoch}")
        try:
            val_losses = 0.0
            mse_loss = 0.0
            ssim = 0.0
            self.net.eval()
            # for idx, batch in enumerate(tqdm(self.dataloader)):
            #     path = batch.pop('path')
            #     batch = self.todevice(batch)
            #     img = batch.pop('img')
            #     mask = batch.pop('mask')
            #     pred = self.net(img)
            #     losses = self.loss_fn(pred, img, mask)
            #     loss = losses["loss"]
            #     losses += loss.detach().item()
            #     if torch.isinf(loss) or torch.isnan(loss):
            #         print(loss)
            #         print(path)
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(self.val_loader)):
                    path = batch.pop('path')
                    batch = self.todevice(batch)
                    img = batch.pop('img')
                    mask = batch.pop('mask')
                    pred = self.net(img)
                    losses = self.loss_fn(pred, img, mask)
                    loss = losses["loss"]
                    val_losses += loss.detach().item()
                    mse_loss += losses["mse"].detach().item()
                    ssim += losses["ssim"].detach().item()
                    if torch.isinf(loss) or torch.isnan(loss):
                        print(loss)
                        print(path)
                    if progress_view:
                        show_visual_progress(img, pred, self.progress_path, title=title, loss=loss)
                        progress_view = False
                    # del loss, pred
                    # torch.cuda.empty_cache()
            # with torch.no_grad():
            logging.info(f"Validation mse: {mse_loss / len(self.val_loader)}, ssim:{ssim / len(self.val_loader)}")
            self.val_loss = val_losses / len(self.val_loader)
        except Exception as e:
            logging.error(f"Exception : {e}")


class TrainSleceNet(Train):
    def __init__(self, net, dataloader, test_dataloader, config, run_paths, device, is_cuda=False):
        super().__init__(net=net, config=config, run_paths=run_paths, is_cuda=is_cuda, model_type="s2lece")
        self.device = device
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        steps_per_epoch = len(self.dataloader)
        self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                           lr=self.learning_rate,
                                           eps=float(config["s2lece"]["epsilon"]))
        # weight_decay=float(config["s2lece"]["weight_decay"])
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
        #                                                      max_lr=self.learning_rate, epochs=self.epochs,
        #                                                      steps_per_epoch=steps_per_epoch,
        #                                                      pct_start=0.05, cycle_momentum=False,
        #                                                      anneal_strategy='linear')
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer,
        #                                                    base_lr=self.learning_rate,
        #                                                    max_lr=self.learning_rate * 10, step_size_up=5,
        #                                                    step_size_down=10, cycle_momentum=False)
        checkpoint = self.load_checkpoint(self.ckpt_path)
        if checkpoint is None:
            self.start_epoch = 0
            logging.info("fresh model....")
        else:
            logging.info("checkpoint model...")
            self.start_epoch = checkpoint["epoch"]
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.loss = 0.0
        self.epe_loss = 0.0
        self.rmse_loss = 0.0
        self.aae_loss = 0.0
        self.diff_loss = 0.0
        self.ssim_loss = 0.0
        self.psnr_loss = 0.0
        self.mse_loss = 0.0
        self.mask_loss = 0.0

    def reset(self):
        self.loss = 0.0
        self.epe_loss = 0.0
        self.rmse_loss = 0.0
        self.aae_loss = 0.0
        self.diff_loss = 0.0
        self.ssim_loss = 0.0
        self.psnr_loss = 0.0
        self.mse_loss = 0.0
        self.mask_loss = 0.0

    def prediction_step(self, inputs, i_epoch=0, progress_view=False):
        inputs = self.todevice(inputs)
        img1 = inputs.pop('img1')
        img2 = inputs.pop('img2')
        target_flow = inputs.pop('flow')
        mask1 = inputs.pop('mask1')
        pred_flow = self.net(img1, img2)
        pred_flow *= mask1
        flow_loss, metrics = flow_loss_fn(img1, img2, target_flow, pred_flow, mask1)

        self.epe_loss += metrics["flow"]["epe loss"].detach().item()
        self.rmse_loss += metrics["flow"]["rmse loss"].detach().item()
        self.aae_loss += metrics["flow"]["aae loss"].detach().item()
        self.diff_loss += metrics["reconstruct"]["diff loss"].detach().item()
        self.ssim_loss += metrics["reconstruct"]["ssim loss"].detach().item()
        self.psnr_loss += metrics["reconstruct"]["psnr loss"].detach().item()
        self.mse_loss += metrics["reconstruct"]["mse loss"].detach().item()
        self.mask_loss += metrics["reconstruct"]["mask loss"].detach().item()

        if progress_view:
            compare_flow(target_flow, pred_flow, self.progress_path, loss=flow_loss, idx=i_epoch)
        return flow_loss

    def train_step(self, epoch):
        self.net.train()
        train_losses = 0.0
        for _, inputs in enumerate(tqdm(self.dataloader)):
            flow_loss = self.prediction_step(inputs)
            assert not (torch.isinf(flow_loss) or torch.isnan(flow_loss)), "Train Loss value is NaN"
            self.optimizer.zero_grad()
            flow_loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            train_losses += flow_loss.detach().item()
            del flow_loss
            torch.cuda.empty_cache()

        before_lr = self.optimizer.param_groups[0]["lr"]
        after_lr = self.optimizer.param_groups[0]["lr"]
        logging.info("Epoch %d: optimizer lr %.6f -> %.6f" % (epoch, before_lr, after_lr))
        self.loss = train_losses / len(self.dataloader)
        result = {"epoch": epoch,
                  "training loss": self.loss,
                  "epe loss": self.epe_loss / len(self.dataloader),
                  "aae loss": self.aae_loss / len(self.dataloader),
                  "rmse loss-flow": self.rmse_loss / len(self.dataloader),
                  "diff loss": self.diff_loss / len(self.dataloader),
                  "ssim loss": self.ssim_loss / len(self.dataloader),
                  "psnr loss": self.psnr_loss / len(self.dataloader),
                  "mse loss": self.mse_loss / len(self.dataloader),
                  "mask loss": self.mask_loss / len(self.dataloader),
                  "learning rate": after_lr}
        self.reset()
        return result

    def validate_step(self, i_epoch, progress_view=True):
        logging.info(f"evaluation at epoch: {i_epoch}")
        val_losses = 0.0
        self.net.eval()
        with torch.no_grad():
            for idx, inputs in enumerate(tqdm(self.test_dataloader)):
                if progress_view:
                    flow_loss = self.prediction_step(inputs, i_epoch=i_epoch, progress_view=progress_view)
                    progress_view = False
                else:
                    flow_loss = self.prediction_step(inputs)
                assert not (torch.isinf(flow_loss) or torch.isnan(flow_loss)), "Val Loss value is NaN"
                val_losses += flow_loss.detach().item()
            self.loss = val_losses / len(self.test_dataloader)

        result = {"validation loss": self.loss,
                  "validation - epe loss": self.epe_loss / len(self.test_dataloader),
                  "validation - aae loss": self.aae_loss / len(self.test_dataloader),
                  "validation - rmse loss-flow": self.rmse_loss / len(self.test_dataloader),
                  "validation - diff loss": self.diff_loss / len(self.test_dataloader),
                  "validation - ssim loss": self.ssim_loss / len(self.test_dataloader),
                  "validation - psnr loss": self.psnr_loss / len(self.test_dataloader),
                  "validation - mse loss": self.mse_loss / len(self.test_dataloader),
                  "validation - mask loss": self.mask_loss / len(self.test_dataloader)
                  }
        self.reset()
        return result


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
