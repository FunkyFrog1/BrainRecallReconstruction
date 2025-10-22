import torch
import torchvision
from diffusers import AutoencoderKL, \
    StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image

class _SDXL():
    def __init__(self):
        vae = AutoencoderKL.from_pretrained("../../vision_backbone/SDXL/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "../../vision_backbone/SDXL/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        self.pipe.load_ip_adapter('../../vision_backbone/SDXL/IP-Adapter', subfolder="sdxl_models",
                                  weight_name="ip-adapter_sdxl.safetensors")

        self.device = self.pipe.device

        seeds = range(1)
        self.generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]


    def generate(self, eeg_z, image):

        prompt = (
            "photograph, natural lighting, realistic, detailed, high resolution, 8k, "
            "sharp focus, professional photography, natural colors, authentic, "
            "lifelike, true to life, unedited, raw photo"
        )

        negative_prompt = (
            "painting, drawing, artwork, illustration, cartoon, anime, "
            "digital art, concept art, stylized, artistic, "
            "low quality, worst quality, bad quality, blurry, pixelated, "
            "deformed, distorted, unrealistic lighting, "
            "low resolution, bad proportions, extra limbs, watermark, text, "
            "saturated, oversaturated, vignette, filter, photoshopped"
        )

        for index in range(eeg_z.shape[0]):
            z = eeg_z[index].unsqueeze(0).unsqueeze(0)
            neg_z = torch.zeros_like(z)

            batch_z = z.repeat(len(self.generators), 1, 1, 1)
            batch_neg_z = neg_z.repeat(len(self.generators), 1, 1, 1)

            results = self.pipe(
                prompt=[prompt] * len(self.generators),
                negative_prompt=[negative_prompt] * len(self.generators),
                image=image,
                ip_adapter_image_embeds=[torch.cat([batch_neg_z, batch_z], dim=0).to(torch.float16)],
                height=512,
                width=512,
                generator=self.generators,
                do_classifier_free_guidance=True,
                num_images_per_prompt=1,
                output_type="pt"
            )

            grid = torchvision.utils.make_grid(results.images)
            torchvision.utils.save_image(grid, 'gen.png')

class SDXL():
    def __init__(self):
        torch.set_grad_enabled(False)
        vae = AutoencoderKL.from_pretrained(
            "../../vision_backbone/SDXL/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "../../vision_backbone/SDXL/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        self.pipe.load_ip_adapter(
            '../../vision_backbone/SDXL/IP-Adapter',
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.safetensors"
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.device = self.pipe.device

        seeds = range(1)
        self.generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    @torch.no_grad()
    def generate(self, eeg_z, image, return_grid=True):
        eeg_z = eeg_z.detach()
        image = image.resize((512, 512))

        prompt = (
            "photograph, natural lighting, realistic, detailed, high resolution, 8k, "
            "sharp focus, professional photography, natural colors, authentic, "
            "lifelike, true to life, unedited, raw photo"
        )

        negative_prompt = (
            "painting, drawing, artwork, illustration, cartoon, anime, "
            "digital art, concept art, stylized, artistic, "
            "low quality, worst quality, bad quality, blurry, pixelated, "
            "deformed, distorted, unrealistic lighting, "
            "low resolution, bad proportions, extra limbs, watermark, text, "
            "saturated, oversaturated, vignette, filter, photoshopped"
        )

        batch_size = len(self.generators)

        for index in range(eeg_z.shape[0]):
            z = eeg_z[index].unsqueeze(0).unsqueeze(0).detach()
            neg_z = torch.zeros_like(z).detach()

            # 批量复制
            batch_z = z.repeat(batch_size, 1, 1, 1)
            batch_neg_z = neg_z.repeat(batch_size, 1, 1, 1)

            # 使用推理模式上下文管理器
            with torch.inference_mode():
                results = self.pipe(
                    prompt=[prompt] * batch_size,
                    negative_prompt=[negative_prompt] * batch_size,
                    image=image,
                    ip_adapter_image_embeds=[torch.cat([batch_neg_z, batch_z], dim=0).to(torch.float16)],
                    height=512,
                    width=512,
                    generator=self.generators,
                    do_classifier_free_guidance=True,
                    guidance_scale=7.5,
                    # num_inference_steps=20,  # 减少步数以加速（根据质量需求调整）
                    num_images_per_prompt=1,
                    output_type="pt",

                    callback=None,  # 禁用回调
                    callback_steps=None  # 禁用回调步数
                )
            # 处理结果张量
            images_tensor = results.images.detach()

            if return_grid:
                grid = torchvision.utils.make_grid(
                    images_tensor,
                )

                return grid.detach()
            else:
                return images_tensor

    def __del__(self):
        # 清理时确保梯度状态正确
        torch.set_grad_enabled(True)


if __name__ == '__main__':
    x = torch.randn((1, 1280))
    img = Image.open('test.jpg').resize((512, 512))
    sdxl = SDXL()
    sdxl.generate(x, img)

