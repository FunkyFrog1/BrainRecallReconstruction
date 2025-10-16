import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ATMS import ATMS as ATMLayer


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


class Ours(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.2):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.visual = nn.Linear(17 * 250, proj_dim)
        self.recall = nn.Linear(self.input_dim, proj_dim)

        self.model = nn.Sequential(
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x_r, x_v):
        B, n, c, d = x_r.shape
        x_v = x_v.view(x_v.shape[0], n, -1)
        x_r = x_r.view(x_r.shape[0], n, self.input_dim)

        x_v = self.visual(x_v)
        x_r = self.recall(x_r)

        # if self.training:
        #     x = x_r
        # else:
        #     x = x_r

        if self.training:
            # 方法7: 使用Gumbel-Softmax进行可微分选择
            logits = torch.randn(3, device=x_r.device)  # 三个选项的logits

            # Gumbel-Softmax采样
            weights = F.gumbel_softmax(logits, tau=1.0, hard=False)

            # 计算混合权重
            alpha = torch.rand(1, device=x_r.device)
            mixed = alpha * x_r + (1 - alpha) * x_v

            # 加权组合
            x = weights[0] * x_r + weights[1] * x_v + weights[2] * mixed
        else:
            x = x_r

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
