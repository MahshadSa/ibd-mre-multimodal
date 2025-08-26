import torch, torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        rn18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_channels != 3:
            w = rn18.conv1.weight.data
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if in_channels == 1:
                with torch.no_grad(): new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
            rn18.conv1 = new_conv
        self.stem   = nn.Sequential(rn18.conv1, rn18.bn1, rn18.relu, rn18.maxpool)
        self.layer1 = rn18.layer1
        self.layer2 = rn18.layer2
        self.layer3 = rn18.layer3
        self.layer4 = rn18.layer4  # -> (B,512,H/32,W/32)

    def forward(self, x):
        x = self.stem(x)
        f4  = self.layer1(x)
        f8  = self.layer2(f4)
        f16 = self.layer3(f8)
        f32 = self.layer4(f16)
        return f4, f8, f16, f32

class ImageBranch(nn.Module):
    def __init__(self, encoder_weights: str, in_channels=1, out_dim=128, freeze=True):
        super().__init__()
        self.backbone = ResNet18Encoder(in_channels=in_channels, pretrained=False)
        state = torch.load(encoder_weights, map_location="cpu")
        self.backbone.load_state_dict(state, strict=True)
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        _, _, _, f32 = self.backbone(x)
        x = self.pool(f32).flatten(1)
        return self.proj(x)   # (B, out_dim)
