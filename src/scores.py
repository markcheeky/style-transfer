import torch
import torch.nn.functional as F


class ContentScore(torch.nn.Module):

    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.score = F.mse_loss(x, self.target)
        return x


class StyleScore(torch.nn.Module):

    @staticmethod
    def gram(x):
        N, C, H, W = x.shape
        assert N == 1
        x = x.reshape(C, H * W)
        return (x @ x.T) / x.numel()

    def __init__(self, target):
        super().__init__()
        self.target = self.gram(target).detach()

    def forward(self, x):
        self.score = F.mse_loss(self.gram(x), self.target)
        return x
