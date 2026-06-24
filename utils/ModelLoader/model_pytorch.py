import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if downsample or stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class FaceRecognitionBackBone(nn.Module):
    def __init__(self, num_channel=3, output_channel=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(output_channel)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        return x


class FaceRecognitionProjector(nn.Module):
    def __init__(self, in_channel=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3 , bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = 64
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * Bottleneck.expansion, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, out_dim)
        self.bn_fc3 = nn.BatchNorm1d(out_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        downsample = stride != 1 or self.inplanes != planes * Bottleneck.expansion
        layers.append(Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        return x


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_channel=3, out_dim=128, dropout=0.1):
        super().__init__()
        # self.backbone = FaceRecognitionBackBone(num_channel, 128, dropout)
        self.projector = FaceRecognitionProjector(num_channel, out_dim, dropout)

    def forward(self, x):
        # x = self.backbone(x)
        x = self.projector(x)
        return x


def load_model(weights_path: str, device=None) -> FaceRecognitionModel:
    model = FaceRecognitionModel(num_channel=3, out_dim=128, dropout=0.1)
    state_dict = torch.load(weights_path, map_location=device , weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and len(state_dict) > 1:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from '{weights_path}' on {device}")
    return model


