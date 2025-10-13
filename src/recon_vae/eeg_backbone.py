import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

import models.MoE
from models.ATMS import ATMS as ATMLayer
import torch.nn.functional as F

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class EEGProject(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.2):
        super(EEGProject, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.GELU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


from models.Conv import OptimizedConvBlock
class Ours(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.2):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.GELU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        B, n, c, d = x.shape
        x = x.view(x.shape[0], n, self.input_dim)
        # if not self.training:
        #     x = self.random_sample_mix(x)
        x = self.model(x)
        if not self.training:
            x = self.random_sample_mix(x)
        x = x.mean(dim=1)
        return x

    def random_mix_augmentation_vectorized(self, x, mix_ratio=0.1, mix_prob=0.5):
        """
        向量化版本的每个样本混合增强（更高效）
        """
        B, n, c, d = x.shape

        if torch.rand(1).item() > mix_prob or n <= 1:
            return x

        # 扩展x以便后续计算 [B, n, n, c, d]
        x_expanded = x.unsqueeze(2).expand(B, n, n, c, d)

        # 创建掩码，排除对角线（自己）
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mask = mask.expand(B, n, n, c, d)

        # 应用掩码获取其他样本 [B, n, n-1, c, d]
        others = x_expanded[mask].view(B, n, n - 1, c, d)

        # 生成随机权重 [B, n, n-1, c, d]
        random_weights = torch.rand(B, n, n - 1, c, d, device=x.device) * mix_ratio

        # 计算每个主样本对应的权重和
        weights_sum = random_weights.sum(dim=2, keepdim=True)  # [B, n, 1, c, d]

        # 归一化权重
        normalized_weights = random_weights / (1 + weights_sum)

        # 计算混合贡献
        mix_contribution = (others * normalized_weights).sum(dim=2)  # [B, n, c, d]

        # 计算主样本权重
        main_weight = 1 - normalized_weights.sum(dim=2)  # [B, n, c, d]

        # 应用混合
        x_aug = x * main_weight + mix_contribution

        return x_aug

    # 简洁版本 - 推荐使用
    def random_sample_mix(self, x, mix_strength=0.05):
        """
        简洁版的每个样本混合增强

        参数:
            x: 输入张量 [B, n, input_dim] 或 [B, n, c, d]
            mix_strength: 混合强度
        """
        B, n, *dims = x.shape
        original_shape = x.shape

        if n <= 1:
            return x

        # 重塑为 [B, n, feature_dim]
        if len(dims) > 1:
            x_flat = x.view(B, n, -1)
        else:
            x_flat = x

        # 复制原始数据
        x_aug = x_flat.clone()

        # 为每个样本生成随机混合掩码
        mix_mask = torch.rand(B, n, 1, device=x.device) < mix_strength

        # 为需要混合的样本生成随机其他样本索引
        other_indices = torch.randint(0, n - 1, (B, n), device=x.device)
        # 调整索引，跳过自己
        other_indices = (other_indices + torch.arange(n, device=x.device).unsqueeze(0)) % n

        # 获取其他样本特征
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(B, n)
        other_features = x_flat[batch_indices, other_indices]

        # 生成随机混合权重
        mix_weights = torch.rand(B, n, 1, device=x.device) * 0.3 + 0.1  # 0.1-0.4

        # 应用混合
        x_aug = torch.where(
            mix_mask,
            x_flat * (1 - mix_weights) + other_features * mix_weights,
            x_flat
        )

        # 恢复原始形状
        return x_aug.view(original_shape)


from models.MoE import MoELayer
class Ours3(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.GELU(),
                                       MoELayer(proj_dim, proj_dim),
                                       nn.GELU(),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class ATMS(ATMLayer):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class BaseModel(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, embedding_dim=1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x


class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )


class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1400)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )


class EEGnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1248)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )


class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )


if __name__ == '__main__':
    pass
    # model = ATMS(1,2,0.3)