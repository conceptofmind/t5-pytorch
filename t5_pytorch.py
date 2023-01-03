import torch
from torch import nn
import torch.nn.functional as F

import math

from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# residual wrapper

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

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

# feedforward layer

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout), # optional dropout
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 12):
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

# T5 Self Attention

class T5SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 12,
        dim_head = 64,
        causal = False,
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
            heads = heads
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        sim = self.relative_position_bias(sim)

        # mask

        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            sim = sim.masked_fill_(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # combine heads and linear output

        return self.to_out(out)

# T5 Cross Attention

class T5CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        context_dim = None,
        heads = 12,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # self.relative_position_bias = T5RelativePositionBias(
        #     scale = dim_head ** -0.5,
        #     causal = False,
        #     heads = heads
        #     )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask = None, context_mask = None):
        b, n, _, h = *x.shape, self.heads

        kv_input = default(context, x)

        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        #sim = self.relative_position_bias(sim)

        # mask

        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            sim = sim.masked_fill_(~mask, mask_value)

        if context_mask is not None:
            sim = sim.masked_fill_(~context_mask[:, None, :], mask_value)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # combine heads and linear output

        return self.to_out(out)

# T5 Encoder

class T5Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        #max_seq_len,
        depth,
        heads = 12,
        dim_head = 64,
        causal = False,
        mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        #self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layer = nn.ModuleList([])
        for _ in range(depth):
            self.layer.append(nn.ModuleList([
                Residual(PreNorm(dim, T5SelfAttention(dim = dim, heads = heads, dim_head = dim_head, causal = causal, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim = dim, mult = mlp_mult, dropout = dropout))),
            ]))

        self.final_norm = T5LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.token_emb(x)
        #x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))

        for attn, mlp in self.layer:
            x = attn(x, mask = mask)
            x = mlp(x)

        x = self.final_norm(x)

        return x

# T5 Decoder

class T5Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        #max_seq_len,
        depth,
        heads = 12,
        dim_head = 64,
        causal = True,
        mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        #self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layer = nn.ModuleList([])
        for _ in range(depth):
            self.layer.append(nn.ModuleList([
                Residual(PreNorm(dim, T5SelfAttention(dim = dim, heads = heads, dim_head = dim_head, causal = causal, dropout = dropout))),
                Residual(PreNorm(dim, T5CrossAttention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim = dim, mult = mlp_mult, dropout = dropout))),
            ]))

        self.final_norm = T5LayerNorm(dim)

    def forward(self, x, context, mask = None, context_mask = None):
        x = self.token_emb(x)
        #x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))

        for attn, cross_attn, mlp in self.layer:
            x = attn(x, mask = mask)
            x = cross_attn(x, context = context, mask = mask, context_mask = context_mask)
            x = mlp(x)

        x = self.final_norm(x)

        return x

# T5

class T5(nn.Module):
    def __init__(
        self,
        *,
        dim,
        #max_seq_len,
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
        dropout = 0.,
        tie_token_emb = True
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(enc_num_tokens, dim)
        #self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.encoder = T5Encoder(
            dim = dim,
            #max_seq_len = max_seq_len, 
            num_tokens = enc_num_tokens, 
            depth = enc_depth, 
            heads = enc_heads, 
            dim_head = enc_dim_head, 
            mlp_mult = enc_mlp_mult, 
            dropout = dropout
        )
        
        self.decoder = T5Decoder(
            dim = dim,
            #max_seq_len= max_seq_len, 
            num_tokens = dec_num_tokens, 
            depth = dec_depth, 
            heads = dec_heads, 
            dim_head = dec_dim_head, 
            mlp_mult = dec_mlp_mult, 
            dropout = dropout
        )

        self.to_logits = nn.Linear(dim, dec_num_tokens)

        # tie weights
        if tie_token_emb:
            self.encoder.token_emb.weight = self.decoder.token_emb.weight

    def forward(self, src, tgt, mask = None, context_mask = None):
        x = self.embedding(src)
        #x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))
        x = self.encoder(src, mask = mask)
        x = self.decoder(tgt, x, mask = mask, context_mask = context_mask)
        x = self.to_logits(x)
        return x


if __name__ == '__main__':
    
    model = T5(
        dim = 768,
        #max_seq_len = 1024,
        enc_num_tokens = 512,
        enc_depth = 6,
        enc_heads = 12,
        enc_dim_head = 64,
        enc_mlp_mult = 4,
        dec_num_tokens = 512,
        dec_depth = 6,
        dec_heads = 12,
        dec_dim_head = 64,
        dec_mlp_mult = 4,
        dropout = 0.,
        tie_token_emb = True
    )

    src = torch.randint(0, 512, (1, 1024))
    src_mask = torch.ones_like(src).bool()
    tgt = torch.randint(0, 512, (1, 1024))

    loss = model(src, tgt, mask = src_mask)

    print(loss.shape) #torch.Size([1, 1024, 512])