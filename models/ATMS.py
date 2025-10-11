import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from models.Embed import DataEmbedding
from models.SelfAttention_Family import FullAttention, AttentionLayer
from models.Transformer_EncDec import Encoder, EncoderLayer


class Config:
    def __init__(self, input_dim):
        self.task_name = 'classification'  # Example task name
        self.seq_len =  input_dim  # Sequence length
        self.pred_len =  250  # Prediction length
        self.output_attention = False  # Whether to output attention weights
        self.d_model =  250  # Model dimension
        self.embed = 'timeF'  # Time encoding method
        self.freq = 'h'  # Time frequency
        self.dropout = 0.25  # Dropout rate
        self.factor = 1  # Attention scaling factor
        self.n_heads = 4  # Number of attention heads
        self.e_layers = 1  # Number of encoder layers
        self.d_ff = 256  # Feedforward network dimension
        self.activation = 'gelu'  # Activation function
        self.enc_in = 63  # Encoder input dimension (example value)


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=1):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]
        # print("enc_out", enc_out.shape)
        return enc_out


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, c_num=63):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num+1, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, c_num=63, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, c_num=c_num),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1280, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    def __init__(self, z_dim, c_num, timesteps):
        super(ATMS, self).__init__()
        input_dim = timesteps[1] - timesteps[0]
        default_config = Config(input_dim)
        self.encoder = iTransformer(default_config)
        self.enc_eeg = Enc_eeg(c_num=c_num)
        self.proj_eeg = Proj_eeg(proj_dim=z_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()


    def forward(self, x):
        subject_ids = torch.zeros((x.shape[0])).long().to(x.device)
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out


if __name__ == '__main__':
    model = ATMS()
    x = torch.randn((4, 17, 250))
    output = model(x)
    print(output.shape)
