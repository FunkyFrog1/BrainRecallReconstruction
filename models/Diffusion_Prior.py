import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset


class DiffusionPrior(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            cond_dim=42,
            hidden_dim=1024,
            layers_per_block=4,
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.time_embedding = TimestepEmbedding(
            time_embed_dim,
            hidden_dim,
        )

        # 2. conditional embedding
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
        )

        # 3.2 hidden
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                )
                for _ in range(layers_per_block)
            ]
        )

        # 3.3 output
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t)  # (batch_size, time_embed_dim)
        t = self.time_embedding(t)  # (batch_size, hidden_dim)

        # 2. conditional embedding
        c = self.cond_embedding(c) if c is not None else 0  # (batch_size, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x)

        # 3.2 hidden
        for layer in self.hidden_layers:
            x = x + t + c
            x = layer(x) + x

        # 3.3 output
        x = self.output_layer(x)

        return x


class DiffusionPriorUNet(nn.Module):

    def __init__(
            self,
            embed_dim=1280,
            cond_dim=42,
            hidden_dim=[1280, 512, 256, 128, 64],
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 2. conditional embedding
        # to 3.2, 3,3

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 hidden encoder
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers - 1)]
        )  # d_0, ..., d_{n-1}
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1)]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                nn.LayerNorm(hidden_dim[i + 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1)]
        )

        # 3.3 hidden decoder
        self.decode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers - 1, 0, -1)]
        )  # d_{n}, ..., d_1
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1, 0, -1)]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i - 1]),
                nn.LayerNorm(hidden_dim[i - 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1, 0, -1)]
        )

        # 3.4 output
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t)  # (batch_size, time_embed_dim)

        # 2. conditional embedding
        # to 3.2, 3.3

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x)

        # 3.2 hidden encoder
        hidden_activations = []
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t)
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)

        # 3.3 hidden decoder
        for i in range(self.num_layers - 1):
            t_emb = self.decode_time_embedding[i](t)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1 - i]

        # 3.4 output
        x = self.output_layer(x)

        return x

        idx_c = self.label2index[self.labels[idx]]
        return {
            "c_embedding": self.embedding_vise[idx_c],
            "h_embedding": self.image_features[idx]
        }


# Copied from diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler.add_noise
def add_noise_with_sigma(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
) -> torch.FloatTensor:
    # Make sure sigmas and timesteps have the same device and dtype as original_samples
    sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
        # mps does not support float64
        schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

    step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigma
    return noisy_samples, sigma


# diffusion pipe
class EmbedDiffusion(nn.Module):
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.embed_dim = embed_dim

        # 扩散先验模型
        self.diffusion_prior = DiffusionPriorUNet(cond_dim=1280, dropout=0.1)

        # 扩散调度器
        from diffusers.schedulers import DDPMScheduler
        self.scheduler = DDPMScheduler()

        # 损失函数
        self.criterion = nn.MSELoss(reduction='none')

        self.device = 'cuda'

    def forward(self, c_embeds=None, h_embeds=None, guidance_drop_prob=0.1, use_gen=False,
                num_inference_steps=10, guidance_scale=2.0, generator=None):

        if use_gen:
            # 使用generate方法进行完整生成
            if c_embeds is None:
                raise ValueError("使用generate方法时需要提供条件嵌入 c_embeds")

            generated_embeds = self.generate(
                c_embeds=c_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            # 计算损失
            loss = self.criterion(generated_embeds, h_embeds)

            return generated_embeds, loss.mean()

        else:
            raw_embeds = c_embeds
            if self.training:
                # 随机条件丢弃
                if c_embeds is not None and torch.rand(1) < guidance_drop_prob:
                    c_embeds = None

            # 采样随机时间步
            num_train_timesteps = self.scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, num_train_timesteps, (h_embeds.shape[0],), device=h_embeds.device)

            noise = torch.randn_like(h_embeds)

            # 添加噪声到目标嵌入
            perturbed_h_embeds = self.scheduler.add_noise(h_embeds, noise, timesteps)

            # 预测噪声
            noise_pred = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)

            # 计算损失
            loss = self.criterion(noise_pred, noise)
            loss = loss.mean()  # 确保正确的reduction

            return raw_embeds, loss

    def generate(
            self,
            c_embeds=None,
            num_inference_steps=10,
            timesteps=None,
            guidance_scale=2.0,
            generator=None
    ):
        self.device = c_embeds.device
        # c_embeds (batch_size, cond_dim)
        self.diffusion_prior.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)

        # 3. Prepare noise
        h_t = torch.randn(N, self.diffusion_prior.embed_dim, generator=generator, device=self.device)

        # 4. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t = torch.ones(h_t.shape[0], dtype=torch.float, device=self.device) * t
            # 4.1 noise prediction
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.diffusion_prior(h_t, t)
            else:
                noise_pred_cond = self.diffusion_prior(h_t, t, c_embeds)
                noise_pred_uncond = self.diffusion_prior(h_t, t)
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = self.scheduler.step(noise_pred, t[0].long().item(), h_t, generator=generator).prev_sample

        return h_t


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    prior = EmbedDiffusion(cond_dim=1024)
    h = torch.randn(2, 1024)
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 1024)
    y, _ = prior(c, h, t)
    print(y.shape)


