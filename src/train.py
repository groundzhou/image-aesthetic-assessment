#!/home/ground/Documents/python/torch-venv/bin/python3

import os

import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from src.trainer import Trainer
from src.model import Nima
from src.loss import EMDLoss
from src.dataset import AVADataset
from src.utils import train_transform, val_transform


def get_data_loaders(csv_dir, image_dir, batch_size, num_workers):
    train_ds = AVADataset(os.path.join(csv_dir, 'train.csv'), image_dir, train_transform)
    val_ds = AVADataset(os.path.join(csv_dir, 'val.csv'), image_dir, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader


def main():
    base_model = mobilenet_v2(pretrained=True)
    model = Nima(base_model, in_features=1280, dropout=0.75)

    optimizer = torch.optim.SGD(
        [{'params': model.features.parameters(), 'lr': 3e-7},
         {'params': model.classifier.parameters(), 'lr': 3e-6}],
        momentum=0.9
    )

    train_data_loader, val_data_loader = get_data_loaders('../data', '../data/images', 3000, 1)

    trainer = Trainer(
        model,
        criterion=EMDLoss(),
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5),
        log_dir='../logs',
        checkpoint_dir='../checkpoints',
        num_epochs=3
    )
    trainer.train()


if __name__ == '__main__':
    main()
