import torch
from tqdm import tqdm
import wandb
import torch.optim.lr_scheduler as toptim
from utils import pytorch_ssim
from visualization.visualization import compare_flow, show_visual_progress
import os
import logging
from models.loss import flow_loss_fn, autoencoder_loss_fn

torch.autograd.set_detect_anomaly(True)


class Train:
    """Base class for training"""
    def __init__(self, net, config, run_paths, dataset_name, model_type, is_cuda=False):
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
            "dataset": dataset_name,
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

                logging.info(log)
                if self.enable_wandb:
                    wandb.log(result)

                if result['validation loss'] < self.best_loss or i == 0:
                    self.best_loss = result['validation loss']
                    logging.info(f"best loss: {self.best_loss}")
                    torch.save({
                        'epoch': i,
                        'state_dict': self.net.state_dict(),
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

                if (i % self.ckpt_interval == 0) or (i == self.epochs-1):
                    logging.info(f"Saving checkpoint at epoch {i}.")
                    torch.save({
                        'epoch': i,
                        'state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': result['training loss'],
                        'val loss': result['validation loss']
                    }, os.path.join(self.ckpt_path, f"ckpts_{i}.pt"))

            # self.net.train()
            if self.model_type == "autoencoder":
                torch.save({'net': 'AutoEncoder', 'state_dict': self.net.state_dict()}, self.save_path)
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
            checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))
        else:
            logging.info("No checkpoint files found.")
        return checkpoint


class TrainAutoEncoder(Train):
    """Class to train Autoencoder"""
    def __init__(self, net, train_loader, val_loader, run_paths, config=None, title=None, is_cuda=False,
                 max_count=0):
        super().__init__(net=net, config=config, run_paths=run_paths, is_cuda=is_cuda, model_type="autoencoder",
                         dataset_name="kitti")
        self.dataloader = train_loader
        self.test_dataloader = val_loader
        self.l2_lambda = config["autoencoder"]["wd_lambda"]
        steps_per_epoch = len(self.dataloader)
        self.weight_decay = config["autoencoder"]["weight_decay"]
        logging.info(self.weight_decay)
        self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                           lr=self.learning_rate,
                                           eps=1e-8, weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=self.learning_rate,
                                                             epochs=self.epochs,
                                                             steps_per_epoch=steps_per_epoch,
                                                             cycle_momentum=False,
                                                             anneal_strategy='linear')

        self.ssim_loss_fn = pytorch_ssim.SSIM(window_size=16)

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
        self.loss = 0.0
        self.ssim_loss = 0.0
        self.psnr_loss = 0.0
        self.diff_loss = 0.0
        self.mask_loss = 0.0
        self.mask_loss = 0.0

    def reset(self):
        self.loss = 0.0
        self.ssim_loss = 0.0
        self.psnr_loss = 0.0
        self.diff_loss = 0.0
        self.mask_loss = 0.0
        self.mask_loss = 0.0

    def prediction_step(self, inputs, i_epoch=0, progress_view=False):
        inputs = self.todevice(inputs)
        inputs = self.todevice(inputs)
        img = inputs.pop('img')
        mask = inputs.pop('mask')
        pred = self.net(img)
        loss, metrics = autoencoder_loss_fn(pred, img, mask)

        self.ssim_loss += metrics["ssim loss"].detach().item()
        self.psnr_loss += metrics["psnr loss"].detach().item()
        self.diff_loss += metrics["diff loss"].detach().item()
        self.mask_loss += metrics["mask loss"].detach().item()

        if progress_view:
            image_title = f'{self.title} - Epoch {i_epoch}'
            show_visual_progress(img, pred, self.progress_path, title=image_title, loss=loss)
        return loss

    def train_step(self, epoch):
        self.net.train()
        train_losses = 0.0
        for _, inputs in enumerate(tqdm(self.dataloader)):
            loss = self.prediction_step(inputs)

            assert not (torch.isinf(loss) or torch.isnan(loss)), "Train Loss value is NaN"
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_losses += loss.detach().item()
            del loss
            torch.cuda.empty_cache()

        before_lr = self.optimizer.param_groups[0]["lr"]
        after_lr = self.optimizer.param_groups[0]["lr"]
        logging.info("Epoch %d: optimizer lr %.6f -> %.6f" % (epoch, before_lr, after_lr))
        self.loss = train_losses / len(self.dataloader)
        result = {"epoch": epoch,
                  "training loss": self.loss,
                  "ssim loss": self.ssim_loss / len(self.dataloader),
                  "psnr loss": self.psnr_loss / len(self.dataloader),
                  "diff loss": self.diff_loss / len(self.dataloader),
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
                    loss = self.prediction_step(inputs, i_epoch=i_epoch, progress_view=progress_view)
                    progress_view = False
                else:
                    loss = self.prediction_step(inputs)
                assert not (torch.isinf(loss) or torch.isnan(loss)), "Val Loss value is NaN"
                val_losses += loss.detach().item()
            self.loss = val_losses / len(self.test_dataloader)

        result = {
            "validation loss": self.loss,
            "validation - ssim loss": self.ssim_loss / len(self.test_dataloader),
            "validation - psnr loss": self.psnr_loss / len(self.test_dataloader),
            "validation - diff loss": self.diff_loss / len(self.test_dataloader),
            "validation - mask loss": self.mask_loss / len(self.test_dataloader)
        }
        self.reset()
        return result


class TrainS2leceNet(Train):
    """Class to train S2lece optical flow network"""
    def __init__(self, net, dataloader, test_dataloader, config, run_paths, device, is_cuda=False):
        super().__init__(net=net, config=config, run_paths=run_paths, is_cuda=is_cuda, model_type="s2lece",
                         dataset_name="kitti")
        self.device = device
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        steps_per_epoch = len(self.dataloader)
        self.weight_decay = float(config["s2lece"]["weight_decay"])
        logging.info(f"weight decay : {self.weight_decay}")
        self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                           lr=self.learning_rate,
                                           eps=float(config["s2lece"]["epsilon"]), weight_decay=self.weight_decay)
        step_size = 5 * steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer,
                                                           base_lr=0.0006442,
                                                           max_lr=self.learning_rate, step_size_up=step_size,
                                                           cycle_momentum=False)
        checkpoint = self.load_checkpoint(self.ckpt_path)
        if checkpoint is None:
            self.start_epoch = 0
            logging.info("fresh model....")
        else:
            logging.info("checkpoint model...")
            self.start_epoch = checkpoint["epoch"]
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

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
        path = inputs.pop('path')
        inputs = self.todevice(inputs)
        img1 = inputs.pop('img1')
        img2 = inputs.pop('img2')
        target_flow = inputs.pop('flow')
        mask1 = inputs.pop('mask1')
        pred_flow = self.net(img1, img2, mask1)
        flow_loss, metrics = flow_loss_fn(img1, img2, target_flow, pred_flow, mask1, max_flow=600)

        self.epe_loss += metrics["flow"]["epe loss"].detach().item()
        self.rmse_loss += metrics["flow"]["rmse loss"].detach().item()
        self.aae_loss += metrics["flow"]["aae loss"].detach().item()
        self.diff_loss += metrics["reconstruct"]["diff loss"].detach().item()
        self.ssim_loss += metrics["reconstruct"]["ssim loss"].detach().item()
        self.psnr_loss += metrics["reconstruct"]["psnr loss"].detach().item()
        self.mse_loss += metrics["reconstruct"]["mse loss"].detach().item()
        self.mask_loss += metrics["reconstruct"]["mask loss"].detach().item()

        if progress_view:
            compare_flow(target_flow, pred_flow[-1], self.progress_path, loss=flow_loss, idx=i_epoch)
        return flow_loss, path

    def train_step(self, epoch):
        self.net.train()
        train_losses = 0.0
        for _, inputs in enumerate(tqdm(self.dataloader)):
            flow_loss, path = self.prediction_step(inputs)
            assert not (torch.isinf(flow_loss) or torch.isnan(flow_loss)), f"Loss value is {flow_loss} at {path}"
            self.optimizer.zero_grad()
            flow_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

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
                    flow_loss, path = self.prediction_step(inputs, i_epoch=i_epoch, progress_view=progress_view)
                    progress_view = False
                else:
                    flow_loss, path = self.prediction_step(inputs)
                assert not (torch.isinf(flow_loss) or torch.isnan(flow_loss)), f"Loss value is {flow_loss} at {path}"
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
