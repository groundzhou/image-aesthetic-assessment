import torch
from torchvision.models import mobilenet_v2
from PIL import Image

from model import Nima, StyleModel
from utils import get_score, val_transform


class InferenceModel:
    """The inference model of image aesthetic assessment.

    Attributes:
        aesthetic_model: Aesthetics assessment.
        style_model: Image style classifier.
    """

    def __init__(self):
        base_model = mobilenet_v2(pretrained=True)
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        # init aesthetic model
        self.aesthetic_model = Nima(base_model, in_features=62720, dropout=0.75)
        self.aesthetic_model.load_state_dict(
            torch.load('../checkpoints/aesthetic-epoch-44.pth', map_location=self.device)['model_state_dict'])
        self.aesthetic_model = self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()

        # init style model
        self.style_model = StyleModel(base_model, in_features=62720, dropout=0.75)
        self.style_model.load_state_dict(
            torch.load('../checkpoints/style-epoch-45.pth', map_location=self.device)['model_state_dict'])
        self.style_model = self.style_model.to(self.device)
        self.style_model.eval()

    def predict(self, image):
        """Evaluate the input image

        Args:
            image(PIL.Image.Image): RGB image input.

        Returns:
            A dict with result. For example:

            {'aesthetic': {'score': 5.701356759760529, 'std': 1.2545808395352676},
             'style': {
                'Complementary_Colors': 0.02640054,
                'Duotones': 0.00874813,
                'HDR': 0.00380639,
                'Image_Grain': 0.0059127 ,
                'Light_On_White': 0.00280869,
                'Long_Exposure': 0.00068638,
                'Macro': 0.11030642,
                'Motion_Blur': 0.00164288,
                'Negative_Image': 0.00186701,
                'Rule_of_Thirds': 0.04127008,
                'Shallow_DOF': 0.18873078,
                'Silhouettes': 0.00798692,
                'Soft_Focus': 0.09419645,
                'Vanishing_Point': 0.00179381}}
        """
        image = val_transform(image)

        image = image.unsqueeze(dim=0)
        image = image.to(self.device)
        with torch.no_grad():
            aesthetic_score = get_score(self.aesthetic_model(image).cpu().numpy()[0])
            styles = self.style_model(image).cpu().numpy()[0]
            return {'aesthetic': dict(zip(['score', 'std'], aesthetic_score)),
                    'style': dict(zip(['Complementary_Colors',
                                       'Duotones'
                                       'HDR',
                                       'Image_Grain',
                                       'Light_On_White',
                                       'Long_Exposure',
                                       'Macro',
                                       'Motion_Blur',
                                       'Negative_Image',
                                       'Rule_of_Thirds',
                                       'Shallow_DOF',
                                       'Silhouettes',
                                       'Soft_Focus',
                                       'Vanishing_Point'], styles))}


def main():
    # image = Image.open('/home/ground/share/Pictures/wallhaven-j5we85.jpg')
    image = Image.open('../data/images/954187.jpg')
    m1 = InferenceModel()
    print(m1.predict(image))


if __name__ == '__main__':
    main()
