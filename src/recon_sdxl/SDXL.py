import torch
from diffusers import AutoencoderKL, \
    StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image

class SDXL():
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

        seeds = range(10)
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
            )

            for index, img in enumerate(results.images):
                img.save(f'./{index}_test.png')


if __name__ == '__main__':
    x = torch.randn((1, 1280))
    img = Image.open('test.jpg').resize((512, 512))
    sdxl = SDXL()
    sdxl.generate(x, img)

