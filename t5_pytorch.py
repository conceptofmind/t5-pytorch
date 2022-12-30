import torch
from torch import nn
import torch.nn.functional as F

import math

from einops import rearrange

# pre-normalization wrapper
# they use layernorm without bias

class T5LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = T5LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# gated-GELU activation function

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# feedforward layer with gated-GELU activation function

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout), # optional dropout
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos, 
            causal = self.causal, 
            num_buckets = self.num_buckets, 
            max_distance = self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return qk_dots + (bias * self.scale)

# T5 attention

class T5Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False,
        num_buckets = 32,
        max_distance = 128,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.relative_position_bias = T5RelativePositionBias(
            scale = dim_head ** -0.5, 
            causal = causal, 
            num_buckets = num_buckets, 
            max_distance = max_distance, 
            heads = heads
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        dots = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.causal:
            i, j = dots.shape[-2:]

        dots = self.relative_position_bias(dots)

        if mask is not None and self.causal:
            # Causal Mask
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            dots = dots.masked_fill(causal_mask, -torch.finfo(dots.dtype).max)

        elif mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        
# T5 Decoder

class T5Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        heads = 8,
        dim_head = 64,
        causal = True,
        num_buckets = 32,
        max_distance = 128,
        mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(1024, dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, T5Attention(dim = dim, heads = heads, dim_head = dim_head, causal = causal, num_buckets = num_buckets, max_distance = max_distance, dropout = dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = mlp_mult, dropout = dropout)),
            ]))

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device
        pos = torch.arange(n, device = device)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropout(x)

        for attn, mlp, in self.layers:
            x = x + attn(x, mask = mask)
            x = x + mlp(x)

        return x

# T5 Encoder

class T5Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        heads = 8,
        dim_head = 64,
        causal = False,
        num_buckets = 32,
        max_distance = 128,
        mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(1024, dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, T5Attention(dim = dim, heads = heads, dim_head = dim_head, causal = causal, num_buckets = num_buckets, max_distance = max_distance, dropout = dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = mlp_mult, dropout = dropout)),
            ]))

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device
        pos = torch.arange(n, device = device)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropout(x)

        for attn, mlp in self.layers:
            x = x + attn(x, mask = mask)
            x = x + mlp(x)

        return x

# T5

class T5(nn.Module):
    def __init__(
        self,
        *,
        dim,
        enc_num_tokens,
        enc_depth,
        enc_heads,
        enc_dim_head,
        enc_mlp_mult,
        dec_num_tokens,
        dec_depth,
        dec_heads,
        dec_dim_head,
        dec_mlp_mult,
        dropout = 0.
    ):
        super().__init__()
        
        self.encoder = T5Encoder(
            dim = dim, 
            num_tokens = enc_num_tokens, 
            depth = enc_depth, 
            heads = enc_heads, 
            dim_head = enc_dim_head, 
            mlp_mult = enc_mlp_mult, 
            dropout = dropout
        )
        
        self.decoder = T5Decoder(
            dim = dim, 
            num_tokens = dec_num_tokens, 
            depth = dec_depth, 
            heads = dec_heads, 
            dim_head = dec_dim_head, 
            mlp_mult = dec_mlp_mult, 
            dropout = dropout
        )

    def forward(self, src, tgt, mask = None):
        x = self.encoder(src, mask = mask)
        x = self.decoder(tgt, mask = mask)
        return x


if __name__ == '__main__':
    
    model = T5(
        dim = 512,
        enc_num_tokens = 256,
        enc_depth = 6,
        enc_heads = 8,
        enc_dim_head = 64,
        enc_mlp_mult = 4,
        dec_num_tokens = 256,
        dec_depth = 6,
        dec_heads = 8,
        dec_dim_head = 64,
        dec_mlp_mult = 4,
        dropout = 0.
    )

    src = torch.randint(0, 256, (1, 1024))
    src_mask = torch.ones_like(src).bool()
    tgt = torch.randint(0, 256, (1, 1024))

    loss = model(src, tgt, mask = src_mask)
    print(loss.shape) #torch.Size([1, 1024, 512])