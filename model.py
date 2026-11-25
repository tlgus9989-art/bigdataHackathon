"""김시현 5695460: 개선 CNN 제출용. 기존 64x64 모델 가중치와 호환되지 않음.
순수 state_dict weights.pt와 함께 제출. logits를 반환하며 사전학습을 사용하지 않음.
"""
import torch
from torch import nn
from torchvision import transforms

CLASSES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
IMG_SIZE = 128

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ImprovedCNN(nn.Module):
    def __init__(self, n_classes=12, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),      # 128 -> 64
            conv_block(32, 64),     # 64 -> 32
            conv_block(64, 128),    # 32 -> 16
            conv_block(128, 192),   # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def load_model(weights_path, device):
    model = ImprovedCNN(n_classes=len(CLASSES))
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
