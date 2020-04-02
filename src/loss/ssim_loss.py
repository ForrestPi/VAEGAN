import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def gaussian(window_size, sigma=None):
    if sigma is None:
        sigma = window_size//2/3
    L = window_size//2
    x = torch.linspace(-L, L, window_size)
    x = x.pow(2) / (2*sigma**2)
    x = F.softmax(-x, dim=0)
    return x


def create_window(window1D=None, channel=3):
    if window1D is None:
        window1D = gaussian()
    window_size = len(window1D)
    window2D = window1D.view(-1, 1) * window1D.view(1, -1)
    window = window2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def rec_ssim(img1, img2, window_size=11, padding=0, val_range=1, method="lcs"):
    """

    :param img1:
    :param img2:
    :param window_size:
    :param padding:
    :param size_average:
    :param val_range:
    :param method: l->luminance, c->contrast, s-> structure
    :return:
    """
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=padding)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1xmu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1.pow(2), window_size, stride=1, padding=padding) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2.pow(2), window_size, stride=1, padding=padding) - mu2_sq
    sigma12 = F.avg_pool2d(img1*img2, window_size, stride=1, padding=padding) - mu1xmu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    C3 = C2 / 2
    ssim_map = 1
    if "l" in method:
        l = (2*mu1*mu2+C1)/(mu1_sq + mu2_sq+C1)
        ssim_map = ssim_map * l
    if "c" in method and "s" in method:
        cs = (2*sigma12 + C2)/(sigma1_sq+sigma2_sq+C2)
        ssim_map = ssim_map * cs
    elif "c" in method:
        c = torch.sqrt(4*sigma1_sq*sigma2_sq+C2**2)/(sigma1_sq+sigma2_sq+c2)
        ssim_map = ssim_map * c
    elif "s" in method:
        s = (sigma12+C3)/torch.sqrt(sigma1_sq*sigma2_sq+C3**2)
        ssim_map = ssim_map * s
    return ssim_map


def win_ssim(img1, img2, window, padding=0, val_range=1, method="lcs"):
    window = window.to(img1)
    channel = img1.size(1)

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1xmu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1xmu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    C3 = C2 / 2
    ssim_map = 1
    if "l" in method:
        l = (2 * mu1 * mu2 + C1) / (mu1_sq + mu2_sq + C1)
        ssim_map = ssim_map * l
    if "c" in method and "s" in method:
        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = ssim_map * cs
    elif "c" in method:
        c = torch.sqrt(4 * sigma1_sq * sigma2_sq + C2 ** 2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ssim_map * c
    elif "s" in method:
        s = (sigma12 + C3) / torch.sqrt(sigma1_sq * sigma2_sq + C3 ** 2)
        ssim_map = ssim_map * s
    return ssim_map


class SSIMLoss(nn.Module):
    def __init__(self, window="gaussian", method="lcs", padding=0, val_range=1, window_size=11, sigma=None, size_average=False):
        super().__init__()
        self.size_average = size_average
        self.ssim = partial(rec_ssim,
                            window_size=window_size,
                            padding=padding,
                            val_range=val_range,
                            method=method)
        if window == "gaussian":
            win = gaussian(window_size, sigma)
            win = create_window(win)
            self.ssim = partial(win_ssim,
                                window=win,
                                padding=padding,
                                val_range=val_range,
                                method=method)

    def forward(self, img1, img2):
        ssim_loss = 1 - self.ssim(img1, img2)
        if self.size_average:
            ssim_loss = ssim_loss.mean()
        else:
            ssim_loss = ssim_loss.sum()
        return ssim_loss