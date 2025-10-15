import importlib
import math
import random
import subprocess

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if config["target"] == "base._pipeline.StableDiffusionXLPipeline":
        return get_obj_from_str(config["target"]).from_pretrained(
            **config.get("params", dict()) if config.get("params", dict()) else {})
    else:
        print(config['target'])
        return get_obj_from_str(config["target"])(
            **config.get("params", dict()) if config.get("params", dict()) else {})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_config(args, config):
    for key in config.keys():
        if hasattr(args, key):
            if getattr(args, key) != None:
                config[key] = getattr(args, key)
    for key in args.__dict__.keys():
        config[key] = getattr(args, key)
    return config


def get_device(gpu_ids):
    if gpu_ids == 'auto':
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free,temperature.gpu', '--format=csv,noheader,nounits'])
        gpu_info_lines = nvidia_smi_output.decode('utf-8').strip().split('\n')
        gpu_info = []
        for line in gpu_info_lines:
            gpu_data = line.strip().split(', ')
            index, memory_free, temperature = map(int, gpu_data)
            gpu_info.append((index, memory_free, temperature))
        gpu_info.sort(key=lambda x: x[1], reverse=True)

        memeory_rank_num = math.ceil(0.4 * len(gpu_info))
        selected_gpus = gpu_info[:memeory_rank_num]
        selected_gpus.sort(key=lambda x: x[2])
        selected_device = selected_gpus[0][0]
        # device = torch.device(f'cuda:{selected_device}')
    elif gpu_ids == "cpu":
        device = torch.device('cpu')
    else:
        gpu_ids = list(map(int, gpu_ids.split(",")))
        selected_device = gpu_ids[0]
        # device = torch.device(f'cuda:{selected_device}')
    return selected_device


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        total_loss = (image_loss + text_loss) / 2

        return total_loss, logits_per_image


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        total_loss = (image_loss + text_loss) / 2

        return total_loss, logits_per_image


class SNClipLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon  # 噪声容忍参数，通常设为0.1-0.2

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        batch_size = image_features.shape[0]

        # 计算相似度矩阵
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # 创建软标签（噪声容忍标签）
        # 正样本标签：1 - epsilon，负样本标签：epsilon / (batch_size - 1)
        soft_labels_image = torch.full_like(logits_per_image, self.epsilon / (batch_size - 1))
        soft_labels_text = torch.full_like(logits_per_text, self.epsilon / (batch_size - 1))

        # 设置对角线为正样本标签
        soft_labels_image[torch.arange(batch_size), torch.arange(batch_size)] = 1 - self.epsilon
        soft_labels_text[torch.arange(batch_size), torch.arange(batch_size)] = 1 - self.epsilon

        # 计算噪声容忍损失
        def noise_tolerant_cross_entropy(logits, soft_labels):
            # 使用softmax计算概率分布
            probs = F.softmax(logits, dim=-1)
            # 计算交叉熵损失：-sum(soft_label * log(prob))
            loss = -torch.sum(soft_labels * torch.log(probs + 1e-8), dim=-1)
            return loss.mean()

        # 计算图像和文本的损失
        image_loss = noise_tolerant_cross_entropy(logits_per_image, soft_labels_image)
        text_loss = noise_tolerant_cross_entropy(logits_per_text, soft_labels_text)

        total_loss = (image_loss + text_loss) / 2

        return total_loss, logits_per_image


def mixco_data(eeg, img_z, alpha=0.2):
    """
    为EEG数据和对应的图像embedding执行MixUp数据增强

    参数:
        eeg: EEG数据张量，形状为(B, n, c, d)
            B - batch大小
            n - 数据重复次数
            c - 通道数
            d - 时间点/数据长度

        img_z: 图像embedding张量，形状为(B, L)
            L - embedding维度

        labels: 标签张量，形状为(B, num_classes)或(B,)
        alpha: Beta分布的参数，控制混合强度

    返回:
        mixed_eeg: 混合后的EEG数据，形状同输入eeg
        mixed_img_z: 混合后的图像embedding，形状同输入img_z
        mixed_labels: 混合后的标签
        lam: 混合比例系数
    """
    # 1. 生成混合比例系数
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # 2. 创建随机索引排列
    batch_size = eeg.size(0)
    indices = torch.randperm(batch_size).to(eeg.device)

    # 3. 混合EEG数据 - 保持重复结构不变
    mixed_eeg = lam * eeg + (1 - lam) * eeg[indices]

    # 4. 混合图像embedding
    mixed_img_z = lam * img_z + (1 - lam) * img_z[indices]

    return mixed_eeg, mixed_img_z, lam


def mixcut_data_tri(eeg, eeg2, img_z, alpha=1.0, time_ratio=0.5):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size, n, c, d = eeg.shape
    indices = torch.randperm(batch_size).to(eeg.device)

    # 随机选择时间窗口
    t_len = int(d * time_ratio)
    t_start = torch.randint(0, d - t_len + 1, (1,)).item()

    mixed_eeg = eeg.clone()
    mixed_eeg2 = eeg2.clone()
    mixed_eeg[:, :, :, t_start:t_start + t_len] = lam * eeg[:, :, :, t_start:t_start + t_len] + (1 - lam) * eeg[indices,
                                                                                                            :, :,
                                                                                                            t_start:t_start + t_len]
    mixed_eeg2[:, :, :, t_start:t_start + t_len] = lam * eeg2[:, :, :, t_start:t_start + t_len] + (1 - lam) * eeg2[
                                                                                                              indices,
                                                                                                              :, :,
                                                                                                              t_start:t_start + t_len]

    mixed_img_z = lam * img_z + (1 - lam) * img_z[indices]
    return mixed_eeg, mixed_eeg2, mixed_img_z


def marginal_distance_matrix_similarity_loss(x, y, margin=0.1):
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    # 计算余弦相似度矩阵
    # 使用矩阵乘法：S = normalized_x * normalized_x^T，结果形状为 (B, B)
    S_x = torch.mm(x, x.t())  # x 的余弦相似度矩阵
    S_y = torch.mm(y, y.t())  # y 的余弦相似度矩阵

    # 计算两个相似度矩阵的绝对差异
    diff = torch.abs(S_x - S_y)  # 形状: (B, B)

    # 减去边际并应用 ReLU，只保留差异超过边际的部分
    loss_matrix = F.relu(diff - margin)  # 形状: (B, B)

    # 计算所有元素的平均值作为最终损失
    loss = loss_matrix.mean()  # 相当于 (1/B^2) * sum(loss_matrix)

    return loss


def marginal_cosine_similarity_loss(z_prime, f, margin=0.1):
    """
    z_prime: 投影后的特征 [batch_size, h, w, feature_dim] 或 [batch_size, feature_dim, h, w]
    f: 基础模型特征 [batch_size, h, w, feature_dim] 或 [batch_size, feature_dim, h, w]
    margin: 边界参数 m1
    """
    # 确保特征维度一致
    assert z_prime.shape == f.shape

    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(z_prime, f, dim=-1)  # 在特征维度计算

    # 边际余弦损失
    loss_per_element = F.relu(1 - margin - cosine_sim)

    # 平均所有空间位置
    loss = loss_per_element.mean()

    return loss
