import torch
from torchvision.models import mobilenet_v2
from PIL import Image
from utils import val_transform
from model import Nima


class InferenceModel:
    def __init__(self):
        base_model = mobilenet_v2(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Nima(base_model, in_features=62720, dropout=0.75)
        self.model.load_state_dict(
            torch.load('../checkpoints/best_state.pth', map_location=self.device)['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image = val_transform(image)
        image = image.unsqueeze(dim=0)
        image = image.to(self.device)
        with torch.no_grad():
            out = self.model(image)
            return out


def main():
    # image = Image.open('/home/ground/share/Pictures/wallhaven-j5we85.jpg')
    image = Image.open('/home/ground/share/image-aesthetic-assessment/data/images/204786.jpg')
    m = InferenceModel()
    print(m.predict(image))


if __name__ == '__main__':
    main()
