import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride) \
                        if (in_ch != out_ch or stride != 1) else None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return F.relu(out + identity)


class backbone(nn.Module):
    def __init__(self, num_channel=3, output_channel=64, dropout=0.1):
        super().__init__()
        self.conv1         = nn.Conv2d(num_channel, 32, 3, stride=1, padding=1)
        self.bn1           = nn.BatchNorm2d(32)
        self.conv2         = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2           = nn.BatchNorm2d(64)
        self.conv3         = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3           = nn.BatchNorm2d(128)
        self.conv4         = nn.Conv2d(128, output_channel, 3, stride=1, padding=1)
        self.bn4           = nn.BatchNorm2d(output_channel)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_layer(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))        
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        return self.dropout_layer(x)


class projector(nn.Module):
    def __init__(self, in_channel=64, out_dim=128, dropout=0.1):
        super().__init__()
        self.resblock1     = ResBlock(in_channel, 512,  stride=2)
        self.resblock2     = ResBlock(512,  512,  stride=1)
        self.resblock3     = ResBlock(512,  512,  stride=2)
        self.resblock4     = ResBlock(512,  1024, stride=1)
        self.flatten       = nn.Flatten()
        self.fc1           = nn.Linear(1024 * 28 * 28, 512)
        self.bn1           = nn.BatchNorm1d(512)
        self.relu          = nn.ReLU()
        self.fc2           = nn.Linear(512, 512)
        self.bn2           = nn.BatchNorm1d(512)
        self.fc3           = nn.Linear(512, out_dim)
        self.bn3           = nn.BatchNorm1d(out_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.flatten(x)
        x = self.dropout_layer(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout_layer(self.relu(self.bn2(self.fc2(x))))
        x = self.bn3(self.fc3(x))
        return x


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_channel=3, out_dim=128, dropout=0.1):
        super().__init__()
        self.backbone  = backbone(num_channel, 256, dropout)
        self.projector = projector(256, out_dim, dropout)

    def forward(self, x):
        return self.projector(self.backbone(x))



def load_model(weights_path: str, device=None) -> FaceRecognitionModel:
    model = FaceRecognitionModel(num_channel=3, out_dim=128, dropout=0.1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from '{weights_path}' on {device}")
    return model


