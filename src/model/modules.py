import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNLReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kerner_size, stride=1, padding=0, negative_slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kerner_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpsampleNearestCBLR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, scale_factor=2):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.pd = nn.ReplicationPad2d(padding)
        self.cblr = ConvBNLReLU(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.up(x)
        x = self.pd(x)
        x = self.cblr(x)
        return x