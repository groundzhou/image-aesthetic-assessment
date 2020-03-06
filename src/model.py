import torch.nn as nn


class Nima(nn.Module):
    """Nima class

    Attributes:
        features: Features of Nima model.
        classifier: Classifier of Nima model.
    """
    def __init__(self,
                 base_model: nn.Module,
                 in_features: int,
                 dropout: float,
                 num_classes=10):
        """Inits Nima with a base model

        Args:
            base_model: Base model of Nima.
            dropout: Dropout rate.
            in_features: Output size of base model.
            num_classes: Number of classes.
        """
        super(Nima, self).__init__()

        # get base model features
        self.features = base_model.features

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=in_features, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
