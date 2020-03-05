import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import time
from src.model.loss import EMDLoss
from src.model.nima import Nima


class Trainer:
    """Trainer class

    Attributes:
        model: Network.
        criterion: Loss function.
        optimizer: Optimizer.
        data_loader: Training data loader.
        valid_data_loader: Valid data loader.
        lr_scheduler: Learning rate scheduler.
        num_epochs: The number of epochs.
    """
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 data_loader,
                 val_data_loader=None,
                 lr_scheduler=None,
                 num_epochs=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs

    def train_epoch(self, epoch):
        """Train an epoch."""
        self.model.train()  # Set model to training mode
        self.optimizer.step()

        loss = self.criterion(outputs, labels)

    def valid_epoch(self, epoch):
        """Validate after training an epoch."""
        self.model.eval()  # Set model to evaluate mode

    def train(self):
        """Full training."""
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # 每个epoch都有一个训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # 迭代数据.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 零参数梯度
                    optimizer.zero_grad()

                    # 前向
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 后向+仅在训练阶段进行优化
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 深度复制mo
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)
        return model

