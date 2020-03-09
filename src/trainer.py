import os
import time
import logging

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from utils import Metrics, get_mean_score

logging.basicConfig(level=logging.INFO,
                    filename='./logs/trainer.log',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__file__)


class Trainer:
    """Trainer class

    Attributes:
        model: Network.
        criterion: Loss function.
        optimizer: Optimizer.
        train_data_loader: Training data loader.
        val_data_loader: Valid data loader.
        scheduler: Learning rate scheduler.
        num_epochs: The number of epochs.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer: Optimizer,
                 train_data_loader,
                 val_data_loader,
                 test_data_loader,
                 scheduler,
                 log_dir,
                 checkpoint_dir,
                 num_epochs):
        print('cuda:', torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.scheduler = scheduler
        self.num_epochs = num_epochs

        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir)
        self.train_step = 0
        self.val_step = 0
        self.print_freq = 20
        self.start_epoch = 1

    def train(self):
        """Full training."""
        best_loss = float('inf')
        best_state = None

        for e in range(self.start_epoch, self.num_epochs + 1):
            train_loss = self.train_epoch(e)
            val_loss = self.validate_epoch()
            self.scheduler.step(metrics=val_loss)

            self.writer.add_scalar('train/loss', train_loss, global_step=e)
            self.writer.add_scalar('val/loss', val_loss, global_step=e)

            if not best_state or val_loss < best_loss:
                logger.info(f'epoch {e} updated loss from {best_loss} to {val_loss}')
                best_loss = val_loss
                best_state = {'epoch': e,
                              'model_state_dict': self.model.state_dict(),
                              'optimizer_state_dict': self.optimizer.state_dict(),
                              'loss': best_loss}
                torch.save(best_state, os.path.join(self.checkpoint_dir, f'style-epoch-{e}.pth'))

    def train_epoch(self, epoch):
        """Train an epoch."""
        self.model.train()  # Set model to training mode
        losses = Metrics()
        total_iter = len(self.train_data_loader.dataset) // self.train_data_loader.batch_size

        for idx, (x, y) in enumerate(self.train_data_loader):
            s = time.monotonic()

            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(x)

            self.optimizer.zero_grad()
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), x.size(0))

            self.writer.add_scalar('train/current_loss', losses.val, self.train_step)
            self.writer.add_scalar('train/avg_loss', losses.avg, self.train_step)
            self.train_step += 1

            e = time.monotonic()
            if idx % self.print_freq == 0:
                log_time = self.print_freq * (e - s)
                eta = ((total_iter - idx) * log_time) / 60.0
                print(f'Epoch {epoch} [{idx}/{total_iter}], loss={loss:.3f}, time={log_time:.2f}, ETA={eta:.2f}')

        return losses.avg

    def validate_epoch(self):
        """Validate after training an epoch."""
        self.model.eval()  # Set model to evaluate mode
        losses = Metrics()

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_data_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                losses.update(loss.item(), x.size(0))

                self.writer.add_scalar('val/current_loss', losses.val, self.val_step)
                self.writer.add_scalar('val/avg_loss', losses.avg, self.val_step)
                self.val_step += 1

        return losses.avg

    def load_state_dict(self, model_path):
        state = torch.load(model_path, map_location=self.device)
        self.start_epoch = state['epoch'] + 1
        self.model.load_state_dict(state['model_state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def test(self):
        self.model.eval()
        losses = Metrics()
        # accuracy = Metrics()

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.test_data_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)
                losses.update(loss.item(), x.size(0))

                # predict = 1 if get_mean_score(y_pred.cpu().numpy()[0]) > 5 else 0
                # target = 1 if get_mean_score(y.cpu().numpy()[0]) > 5 else 0
                #
                # accuracy.update(1 if predict == target else 0)

        logger.info(f'test loss={losses.avg}')
        print(losses.avg)
        return losses.avg
