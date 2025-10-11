import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable
import math


class Expert(nn.Module):
    """单个专家模块"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: Optional[Callable] = None, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # # 初始化权重
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        # if bias:
        #     nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Router(nn.Module):
    """路由器模块，负责分配输入到不同的专家"""

    def __init__(self, in_features: int, num_experts: int, top_k: int = 1,
                 capacity_factor: float = 1.0, noisy_gating: bool = True):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating

        # 路由权重
        self.weight = nn.Parameter(torch.empty(in_features, num_experts))
        if noisy_gating:
            self.noise_weight = nn.Parameter(torch.empty(in_features, num_experts))

        # 初始化
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))
        if noisy_gating:
            nn.init.normal_(self.noise_weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, in_features]
        Returns:
            gates: [batch_size, seq_len, num_experts]
            indices: [batch_size, seq_len, top_k]
        """
        if self.noisy_gating:
            # 添加噪声
            noise = torch.randn_like(x) @ self.noise_weight
            clean_logits = x @ self.weight
            noisy_logits = clean_logits + noise
            logits = noisy_logits
        else:
            logits = x @ self.weight

        # 计算门控值
        gates = F.softmax(logits, dim=-1)

        # 选择top_k专家
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)

        # 归一化门控值
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        return top_k_gates, top_k_indices


class MoELayer(nn.Module):
    """
    Mixture of Experts层，可以替换Linear层
    """

    def __init__(self, in_features: int, out_features: int, num_experts: int = 8,
                 top_k: int = 2, bias: bool = True, activation: Optional[Callable] = None,
                 dropout: float = 0.0, capacity_factor: float = 1.0,
                 noisy_gating: bool = True, expert_class: type = Expert):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # 创建专家
        self.experts = nn.ModuleList([
            expert_class(in_features, out_features, bias, activation, dropout)
            for _ in range(num_experts)
        ])

        # 路由器
        self.router = Router(in_features, num_experts, top_k, capacity_factor, noisy_gating)

        # 辅助损失
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, in_features] or [batch_size, in_features]
        Returns:
            output: [batch_size, seq_len, out_features] or [batch_size, out_features]
        """
        original_shape = x.shape
        if len(original_shape) == 2:
            # [batch_size, in_features] -> [batch_size, 1, in_features]
            x = x.unsqueeze(1)

        batch_size, seq_len, in_features = x.shape

        # 路由计算
        gates, indices = self.router(x)

        # 计算辅助损失（负载均衡损失）
        self._calculate_aux_loss(gates, indices)

        # 重塑输入以便处理
        x_flat = x.reshape(-1, in_features)  # [batch_size * seq_len, in_features]

        # 初始化输出
        output = torch.zeros(batch_size * seq_len, self.out_features, device=x.device)

        # 为每个top_k专家计算输出
        for i in range(self.top_k):
            # 获取当前专家的门控值和索引
            gate_i = gates[..., i].reshape(-1, 1)  # [batch_size * seq_len, 1]
            indices_i = indices[..., i].reshape(-1)  # [batch_size * seq_len]

            # 为每个专家处理对应的输入
            for expert_idx in range(self.num_experts):
                # 选择分配给当前专家的样本
                mask = indices_i == expert_idx
                if mask.sum() == 0:
                    continue

                # 获取专家输出
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)

                # 加权并累加到输出
                output[mask] += gate_i[mask] * expert_output

        # 恢复原始形状
        output = output.reshape(batch_size, seq_len, self.out_features)

        if len(original_shape) == 2:
            # [batch_size, 1, out_features] -> [batch_size, out_features]
            output = output.squeeze(1)

        return output

    def _calculate_aux_loss(self, gates: torch.Tensor, indices: torch.Tensor):
        """计算负载均衡辅助损失"""
        batch_size, seq_len, _ = gates.shape

        # 计算每个专家的门控值总和
        expert_gates = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.top_k):
            expert_gates.scatter_add_(0, indices[..., i].reshape(-1), gates[..., i].reshape(-1))

        # 计算每个专家的选择概率
        expert_prob = expert_gates / (batch_size * seq_len)

        # 计算专家选择分布的熵
        aux_loss = (expert_prob * torch.log(expert_prob + 1e-9)).sum()

        self.aux_loss = -aux_loss  # 负熵作为损失

    def get_aux_loss(self) -> torch.Tensor:
        """获取辅助损失"""
        return self.aux_loss

    def reset_aux_loss(self):
        """重置辅助损失"""
        self.aux_loss = 0.0


# 使用示例
if __name__ == "__main__":
    # 示例：创建一个简单的Transformer模型并用MoE替换FFN层
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)


    # 创建模型
    model = SimpleTransformer(vocab_size=1000, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048)

    # 替换FFN层为MoE
    replaced_count = MoEReplacer.replace_linear_with_moe(
        model,
        target_layers=["linear1", "linear2"],  # TransformerEncoderLayer中的FFN层
        num_experts=8,
        top_k=2,
        activation=nn.ReLU(),
        dropout=0.1
    )

    print(f"Replaced {replaced_count} layers with MoE")

    # 测试前向传播
    x = torch.randint(0, 1000, (32, 50))  # [batch_size, seq_len]
    output = model(x)
    print(f"Output shape: {output.shape}")
