#!/home/ground/Documents/python/torch-venv/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load('./models/epoch-57.pkl'))
model = model.to(torch.device('cuda'))
model.eval()

test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    ])

image = Image.open('/home/ground/share/pictures/wallhaven-j5we85.jpg')
imt = test_transform(image)
imt = imt.unsqueeze(dim=0)
imt = imt.to(torch.device('cuda'))
with torch.no_grad():
    out = model(imt)
out = out.view(10, 1)
print(out)
