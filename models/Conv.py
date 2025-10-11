import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积，参数更少，效果更好"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, dilation=dilation
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    """修复的残差块，避免inplace操作"""

    def __init__(self, d, kernel_size, dilation=1, use_sep_conv=False):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2

        if use_sep_conv:
            self.conv1 = DepthwiseSeparableConv1d(d, d, kernel_size, padding, dilation)
            self.conv2 = DepthwiseSeparableConv1d(d, d, kernel_size, padding, dilation)
        else:
            self.conv1 = nn.Conv1d(d, d, kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(d, d, kernel_size, padding=padding, dilation=dilation)

        self.norm1 = nn.BatchNorm1d(d)
        self.norm2 = nn.BatchNorm1d(d)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x

        # 第一个卷积层
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 第二个卷积层
        x = self.conv2(x)
        x = self.norm2(x)

        # 残差连接 - 使用加法而不是inplace操作
        x = x + residual  # 这行是安全的
        x = self.activation(x)

        return x


class MultiScaleConv(nn.Module):
    """多尺度卷积，修复inplace问题"""

    def __init__(self, d, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, ks, padding=ks // 2) for ks in kernel_sizes
        ])
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        self.norm = nn.BatchNorm1d(d)

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))

        # 加权融合 - 避免inplace操作
        weights = F.softmax(self.weights, dim=0)
        weighted_outputs = [w * out for w, out in zip(weights, outputs)]
        x = torch.stack(weighted_outputs).sum(dim=0)  # 使用stack和sum而不是inplace加法
        x = self.norm(x)
        return x


# 修复的优化卷积设计
class OptimizedConvBlock(nn.Module):
    def __init__(self, d, c_num, use_multi_scale=False, use_sep_conv=True):
        super().__init__()

        if use_multi_scale:
            # 方案1: 多尺度卷积
            self.conv = nn.Sequential(
                MultiScaleConv(d, kernel_sizes=[max(3, c_num - 2), c_num, c_num + 2]),
                ResidualBlock(d, c_num, use_sep_conv=use_sep_conv),
                ResidualBlock(d, c_num, dilation=2, use_sep_conv=use_sep_conv),
                nn.Conv1d(d, d, c_num, padding=c_num // 2)
            )
        else:
            # 方案2: 渐进式感受野
            self.conv = nn.Sequential(
                ResidualBlock(d, c_num, dilation=1, use_sep_conv=use_sep_conv),
                ResidualBlock(d, c_num, dilation=2, use_sep_conv=use_sep_conv),
                ResidualBlock(d, c_num, dilation=4, use_sep_conv=use_sep_conv),
                nn.Conv1d(d, d, c_num, padding=c_num // 2)
            )

        self.final_norm = nn.BatchNorm1d(d)
        self.final_activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.final_norm(x)
        x = self.final_activation(x)
        return x


# 修复的轻量级版本
class LightweightConvBlock(nn.Module):
    def __init__(self, d, c_num):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv1d(d, d, c_num, padding=c_num // 2),
            nn.BatchNorm1d(d),
            nn.GELU(),
            nn.Dropout(0.1),

            DepthwiseSeparableConv1d(d, d, c_num, padding=c_num // 2),
            nn.BatchNorm1d(d),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv1d(d, d, c_num, padding=c_num // 2)
        )

    def forward(self, x):
        return self.conv(x)


# 最安全的版本 - 直接替换你的原始设计
class SafeConvBlock(nn.Module):
    """最安全的版本，避免所有可能的inplace操作"""

    def __init__(self, d, c_num):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d, d, c_num, padding=c_num // 2),
                nn.BatchNorm1d(d),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


# 兼容你原始设计的版本
class CompatibleConvBlock(nn.Module):
    """完全兼容你原始设计的版本，只是添加了必要的组件"""

    def __init__(self, d, c_num):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, c_num, padding=c_num // 2),
            nn.BatchNorm1d(d),  # 添加BN稳定训练
            nn.GELU(),  # 添加激活函数
            nn.Dropout(0.1),  # 添加dropout

            nn.Conv1d(d, d, c_num, padding=c_num // 2),
            nn.BatchNorm1d(d),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv1d(d, d, c_num, padding=c_num // 2),
            nn.BatchNorm1d(d),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


# 测试修复的版本
if __name__ == "__main__":
    # 启用异常检测来验证修复
    torch.autograd.set_detect_anomaly(True)

    d = 250
    c_num = 5
    seq_len = 17
    batch_size = 32

    # 创建测试输入
    x = torch.randn(batch_size, d, seq_len, requires_grad=True)

    print("测试修复的版本:")

    # 测试所有修复的版本
    modules_to_test = [
        ("SafeConvBlock", SafeConvBlock(d, c_num)),
        ("CompatibleConvBlock", CompatibleConvBlock(d, c_num)),
        ("LightweightConvBlock", LightweightConvBlock(d, c_num)),
        ("OptimizedConvBlock-MultiScale", OptimizedConvBlock(d, c_num, use_multi_scale=True)),
        ("OptimizedConvBlock-Progressive", OptimizedConvBlock(d, c_num, use_multi_scale=False)),
    ]

    for name, module in modules_to_test:
        try:
            print(f"\n测试 {name}:")
            # 前向传播
            output = module(x)
            print(f"前向传播成功, 输出形状: {output.shape}")

            # 反向传播
            loss = output.sum()
            loss.backward()
            print(f"反向传播成功")

            print(f"参数量: {sum(p.numel() for p in module.parameters()):,}")

        except Exception as e:
            print(f"错误: {e}")
