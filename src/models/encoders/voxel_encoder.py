# src/models/voxel_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelEncoder(nn.Module):
    def __init__(self, in_chans: int = 1, width=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = width
        self.conv1 = nn.Conv3d(in_chans, c1, 3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm3d(c1)

        self.conv2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm3d(c2)

        self.conv3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm3d(c3)

        self.conv4 = nn.Conv3d(c3, c4, 3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm3d(c4)

        self.fc    = nn.Linear(c4 * 3 * 3 * 3, 512)
        self.out_dim = 512

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # 48³ → 24³
        x = F.relu(self.bn2(self.conv2(x)))   # 24³ → 12³
        x = F.relu(self.bn3(self.conv3(x)))   # 12³ →  6³
        x = F.relu(self.bn4(self.conv4(x)))   #  6³ →  3³
        x = x.flatten(1)                      # (B, c4*3*3*3)
        x = self.fc(x)                        # (B, 512)
        return x
