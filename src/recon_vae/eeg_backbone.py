import numpy as np
import torch
import torch.nn as nn

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
                                   )), )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.mean(dim=1)
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class Ours(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.7):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.SiLU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim)
                                   )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        self._init_weights()

    def forward(self, x):
        B, n, c, d = x.shape
        x = x.view(B, n, self.input_dim)
        x = self.model(x).mean(dim=1)
        return x

    def _init_weights(self):
        """参数初始化 - 使用Kaiming/He初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化（针对GELU激活函数）
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity='relu')  # GELU在零点附近类似ReLU
                # 偏置项初始化为零
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Ours_bn(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.7):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.SiLU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   )
        self.bn = nn.BatchNorm1d(proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        self._init_weights()

    def forward(self, x):
        B, n, c, d = x.shape
        x = x.view(B, n, self.input_dim)
        x = self.model(x)
        x = x.view(B * n, self.z_dim)
        x = self.bn(x)
        x = x.view(B, n, self.z_dim).mean(dim=1)
        return x

    def _init_weights(self):
        """参数初始化 - 使用Kaiming/He初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化（针对GELU激活函数）
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity='relu')  # GELU在零点附近类似ReLU
                # 偏置项初始化为零
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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
