import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL


class VAEProcessor():
    def __init__(self, vae_path='../../vision_backbone/vae', device='cuda', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=self.dtype
        ).to(device).eval()

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels
        )

        for param in self.vae.parameters():
            param.requires_grad = False

    def encode_to_latent_dist(self, x):
        x = self.image_processor.preprocess(x)

        with torch.no_grad():
            if x.dtype != self.dtype:
                x = x.to(self.dtype)

            posterior = self.vae.encode(x).latent_dist
            latent_sample = posterior.sample()

            return posterior, latent_sample

    def decode_from_latent(self, latent, post_process=True):
        if latent.dtype != self.dtype:
            latent = latent.to(self.dtype)

        decoded = self.vae.decode(latent).sample

        if post_process:
            decoded = self.image_processor.postprocess(
                decoded,
                output_type="pt",
                do_denormalize=[True] * decoded.shape[0])  # 关键修复
        return decoded

    def encode_to_latent(self, x):
        """
        简化的编码接口，只返回潜在变量
        """
        _, latent_sample = self.encode_to_latent_dist(x)
        return latent_sample


# 使用示例
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def test_vae():
    """
    测试VAE处理器功能
    """
    # 初始化处理器
    vae_processor = VAEProcessor()

    try:
        # 加载测试图像
        x = PIL.Image.open('./test.jpg').convert('RGB').resize((32, 32))

        # 预处理变换
        process_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        x = process_transform(x).unsqueeze(0).to(vae_processor.device)

        print(f"原始图像形状: {x.shape}")

        # 1. 编码为潜在分布
        posterior, latent_sample = vae_processor.encode_to_latent_dist(x)
        print(f"潜在变量形状: {latent_sample.shape}")
        print(latent_sample)

        # 2. 从潜变量解码回图像
        reconstructed_image = vae_processor.decode_from_latent(latent_sample, post_process=False)
        print(reconstructed_image)
        print(f"重建图像形状: {reconstructed_image.shape}")

        # 3. 计算重建质量指标
        mse_loss = F.mse_loss(reconstructed_image, x).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse_loss))

        print(f"重建MSE损失: {mse_loss:.6f}")
        print(f"PSNR: {psnr:.2f} dB")

        # 4. 可视化结果
        visualize_comparison(x, reconstructed_image, mse_loss, psnr)

    except FileNotFoundError:
        print("测试图像未找到，使用随机张量进行测试")


def visualize_comparison(original, reconstructed, mse_loss, psnr):
    """
    可视化原始图像和重建图像的对比
    """
    # 转换为numpy数组用于显示
    original_np = original.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    recon_np = reconstructed.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()

    # 裁剪到[0,1]范围
    original_np = np.clip(original_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(original_np)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    # 重建图像
    axes[1].imshow(recon_np)
    axes[1].set_title(f'重建图像\nPSNR: {psnr:.2f}dB')
    axes[1].axis('off')

    # 差异图像
    diff = np.abs(original_np - recon_np)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'差异图像\nMSE: {mse_loss:.6f}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_vae()
