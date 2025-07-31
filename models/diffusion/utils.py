import math
import torch
import torch.nn as nn

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class TimeStepEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        out_features: int = 512,
        drop_rate: float = 0.2
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dense = nn.ModuleList([
            nn.Linear(embedding_dim, out_features),
            # nn.SiLU(),
            # nn.GELU(),
            # nn.Dropout(drop_rate),
            nn.Linear(out_features, out_features),
            # nn.Dropout(drop_rate),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = get_timestep_embedding(x, self.embedding_dim)
        x = self.dense[0](x)
        x = nonlinearity(x)
        x = self.dense[1](x)
        return x