# model_pytorch.py — معماری Python مطابق C++

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) \
                        if (in_ch != out_ch or stride != 1) else None

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
        self.conv1         = nn.Conv2d(num_channel, 64, 3, stride=1, padding=1, bias=False)
        self.bn1           = nn.BatchNorm2d(64)
        self.conv2         = nn.Conv2d(64, output_channel, 3, stride=1, padding=1, bias=False)
        self.bn2           = nn.BatchNorm2d(output_channel)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.dropout_layer(x)


class Projector(nn.Module):
    def __init__(self, in_channel=64, out_dim=128, dropout=0.1):
        super().__init__()
        self.resblock1     = ResBlock(in_channel, 128,  stride=2)
        self.resblock2     = ResBlock(128,  256,  stride=1)
        self.resblock3     = ResBlock(256,  512,  stride=2)
        self.resblock4     = ResBlock(512,  1024, stride=1)
        self.resblock5     = ResBlock(1024, 512,  stride=2)
        self.flatten       = nn.Flatten()
        self.fc1           = nn.Linear(512 * 28 * 28, 512)
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
        return self.projector(self.backbone(x))



def _remap_key(k: str) -> str:

    return k


def load_model(weights_path: str, device=None) -> FaceRecognitionModel:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FaceRecognitionModel(num_channel=3, out_dim=128, dropout=0.1)

    raw = torch.load(weights_path, map_location='cpu', weights_only=False)

    if isinstance(raw, dict):
        remapped = {_remap_key(k): v for k, v in raw.items()}
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    else:
        raise ValueError(
            f"Unexpected type from torch.load: {type(raw)}\n"
            "مطمئن شو export_model.exe با موفقیت اجرا شده و model_weights.pt ساخته شده."
        )

    model.to(device)
    model.eval()
    print(f"✓ Model loaded from '{weights_path}' on {device}")
    return model


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "model_weights.pt"
    m = load_model(path)
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 224, 224))
    print(f"✓ Output shape: {out.shape}") 