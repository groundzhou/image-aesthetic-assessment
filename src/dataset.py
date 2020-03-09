import os

from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class AVADataset(Dataset):
    """AVA dataset"""

    def __init__(self, csv_file, root_dir, transform, type='Nima'):
        """
        Args:
            csv_file（string）：带注释的csv文件的路径。
            root_dir（string）：包含所有图像的目录。
            transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.type = type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        image_id = row['image_id']
        image_path = os.path.join(self.root_dir, '{}.jpg'.format(image_id))
        try:
            image = pil_loader(image_path)
        except:
            return self.__getitem__(item-1)

        x = self.transform(image)
        y = row[1:].values.astype("float32")
        if self.type == 'Nima':
            y = y / y.sum()
        return x, y


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
