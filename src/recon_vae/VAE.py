from diffusers.models.autoencoders import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F
import torch

class VAEProcessor:
    def __init__(self, vae_path='../../vision_backbone/vae', device='cuda'):
        """
        初始化VAE处理器

        参数:
            vae_path: VAE模型路径
            device: 运行设备
        """
        self.device = device
        self.dtype = torch.bfloat16

        # 加载VAE模型
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=self.dtype
        ).to(device).eval()

        # 设置图像处理器
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels
        )

        # 禁用梯度计算
        for param in self.vae.parameters():
            param.requires_grad = False

    def encode_to_latent_dist(self, x):
        with torch.no_grad():
            # 确保输入在正确范围内并转换为正确数据类型
            if x.dtype != self.dtype:
                x = x.to(self.dtype)

            # 如果图像值范围是[0,1]，需要缩放到[-1,1]
            if x.min() >= 0 and x.max() <= 1:
                x = x * 2 - 1

            # 使用VAE编码器
            posterior = self.vae.encode(x).latent_dist
            latent_sample = posterior.sample()

            return posterior, latent_sample

    def decode_from_latent(self, latent, return_pil=False):
        with torch.no_grad():
            # 确保潜变量数据类型正确
            if latent.dtype != self.dtype:
                latent = latent.to(self.dtype)

            # 使用VAE解码器
            decoded = self.vae.decode(latent).sample

            # 将输出从[-1,1]缩放到[0,1]
            decoded = (decoded + 1) / 2

            # 如果需要返回PIL图像
            if return_pil:
                decoded = self.image_processor.postprocess(
                    decoded, output_type="pil", do_denormalize=[True]
                )

            return decoded


# 使用示例
def test_vae():
    # 初始化处理器
    vae_processor = VAEProcessor()

    # 假设你有一张已经resize好的图片 x
    # x 的形状应该是 [B, C, H, W]，值范围最好是[0,1]或[-1,1]
    x = torch.rand(1, 3, 32, 32).cuda()

    # 1. 编码为latent distribution
    posterior, latent_sample = vae_processor.encode_to_latent_dist(x)
    print(latent_sample.shape)

    # 3. 从潜变量解码回图像
    reconstructed_image = vae_processor.decode_from_latent(latent_sample)
    print(reconstructed_image.shape)
    print((x - reconstructed_image).mean())

    # 4. 也可以从分布的均值解码（更稳定的重建）
    # reconstructed_from_mean = vae_processor.decode_from_latent(mean)


if __name__ == '__main__':
    test_vae()