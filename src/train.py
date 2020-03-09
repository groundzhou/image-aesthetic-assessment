#!/home/ground/Documents/python/torch-venv/bin/python3

import os

import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from trainer import Trainer
from model import Nima, StyleModel
from loss import EMDLoss
from dataset import AVADataset
from utils import train_transform, val_transform


def get_data_loaders(csv_dir, image_dir, batch_size, num_workers):
    train_ds = AVADataset(os.path.join(csv_dir, 'train.csv'), image_dir, train_transform)
    val_ds = AVADataset(os.path.join(csv_dir, 'val.csv'), image_dir, val_transform)
    test_ds = AVADataset(os.path.join(csv_dir, 'test.csv'), image_dir, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader


base_model = mobilenet_v2(pretrained=True)
train_data_loader, val_data_loader, test_data_loader = get_data_loaders('../data/images', '../data/images', 2, 2)


def train1():
    model = Nima(base_model, in_features=62720, dropout=0.75)
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 3e-7},
        {'params': model.classifier.parameters(), 'lr': 3e-6}
    ], lr=3e-7)

    aesthetic_trainer = Trainer(
        model,
        criterion=EMDLoss(),
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5),
        log_dir='../logs',
        checkpoint_dir='../checkpoints',
        num_epochs=1
    )
    aesthetic_trainer.load_state_dict('/root/IAA/checkpoints/epoch-44.pth')
    aesthetic_trainer.test()


def train2():
    model = StyleModel(base_model, in_features=62720, dropout=0.75)
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 3e-7},
        {'params': model.classifier.parameters(), 'lr': 3e-6}
    ], lr=3e-7)

    style_trainer = Trainer(
        model,
        criterion=torch.nn.BCELoss(),
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5),
        log_dir='../logs',
        checkpoint_dir='../checkpoints',
        num_epochs=10
    )
    style_trainer.train()
    style_trainer.test()


def main():
    train2()


if __name__ == '__main__':
    main()
