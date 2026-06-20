import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return F.relu(out + identity)


class BackBone(nn.Module):
    def __init__(self, num_channel=3, output_channel=64, dropout=0.1):
        super().__init__()
        self.conv1          = nn.Conv2d(num_channel, 64, 3, stride=1, padding=1, bias=False)
        self.bn1            = nn.BatchNorm2d(64)
        self.conv2          = nn.Conv2d(64, output_channel, 3, stride=1, padding=1, bias=False)
        self.bn2            = nn.BatchNorm2d(output_channel)
        self.dropout_layer  = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.dropout_layer(x)


class Projector(nn.Module):
    def __init__(self, in_channel=64, out_dim=128, dropout=0.1):
        super().__init__()
        self.resblock1      = ResBlock(in_channel, 128, stride=2)
        self.resblock2      = ResBlock(128, 256, stride=1)
        self.resblock3      = ResBlock(256, 512, stride=2)
        self.resblock4      = ResBlock(512, 1024, stride=1)
        self.resblock5      = ResBlock(1024, 512, stride=2)
        self.flatten        = nn.Flatten()
        self.fc1            = nn.Linear(512 * 28 * 28, 512)
        self.bn1            = nn.BatchNorm1d(512)
        self.relu           = nn.ReLU()
        self.fc2            = nn.Linear(512, 512)
        self.bn2            = nn.BatchNorm1d(512)
        self.fc3            = nn.Linear(512, out_dim)
        self.bn3            = nn.BatchNorm1d(out_dim)
        self.dropout_layer  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.flatten(x)
        x = self.dropout_layer(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout_layer(self.relu(self.bn2(self.fc2(x))))
        x = self.bn3(self.fc3(x))
        return x


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_channel=3, out_dim=128, dropout=0.1):
        super().__init__()
        self.backbone  = BackBone(num_channel, 64, dropout)
        self.projector = Projector(64, out_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return x


def load_model(weights_path: str, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FaceRecognitionModel(num_channel=3, out_dim=128, dropout=0.1)

    raw = torch.load(weights_path, map_location='cpu')

    if isinstance(raw, dict):
        model.load_state_dict(raw, strict=False)
    elif isinstance(raw, list):
        state = {k: v for k, v in raw}
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unexpected format: {type(raw)}")

    model.to(device)
    model.eval()
    print(f"Model loaded from {weights_path} on {device}")
    return model


if __name__ == "__main__":
    m = load_model("model_weights.pt")
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = m(dummy)
    print(f"Output shape: {out.shape}")