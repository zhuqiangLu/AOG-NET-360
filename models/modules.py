import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange
from transformers import CLIPVisionConfig 
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
# classes

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states





def eye_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.eye_(m.weight,)


def zero_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)



'''
Copy from transformers, remove positional embed
'''
class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # self.num_patches = (self.image_size // self.patch_size) ** 2
        # self.num_positions = self.num_patches + 1
        # self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return patch_embeds



###########################################
############# differential topK ###########
###########################################
# Calculation of differential topK is based on [Top-K](https://arxiv.org/pdf/2104.03059.pdf), thanks
class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int=500, sigma: float=0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
    
    def __call__(self, x):
        return PerturbedTopKFuntion.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFuntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int=500, sigma: float=0.05):
        # input here is scores with (bs, num_patches)
        b, d = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(dtype=x.dtype, device=x.device)
        perturbed_x = x.unsqueeze(1) + noise*sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = F.one_hot(indices, num_classes=d).float() # b, nS, k, d
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # context for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None]*5)
        
        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None]*5)



class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x



def get_rot(pos, flat=True):

    sin = torch.sin(pos) 
    cos = torch.cos(pos)
    rot_1 = torch.stack([cos, -1*sin], dim=-1) 
    rot_2 = torch.stack([sin, cos], dim=-1) 
    rot = torch.stack([rot_1, rot_2], dim=-1)
    if flat:
        rot = rot.flatten(-3, -1)

    return rot

def get_cartesian(pos, flat=True):
    # pos in shape b h w [phi, theta]
    phi = pos[:, 0, :, :]
    theta = pos[:, 1, :, :]

    x = torch.sin(phi)*torch.cos(theta) 
    y = torch.sin(phi)*torch.sin(theta) 
    z = torch.cos(theta) 

    cartesian = torch.stack([x, y, z], dim=1)

    return cartesian
    


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., linear=False, init_fn=None):
        super().__init__()
        self.linear = linear
        if not linear:
            # self.net = nn.Sequential(
            #     nn.Linear(dim, hidden_dim, bias=False),
            #     nn.GELU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(hidden_dim, dim, bias=False),
            #     nn.Dropout(dropout)
            # )
            self.net = nn.ModuleList([
                nn.Linear(dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim, bias=False),
                nn.Dropout(dropout),
            ])
        else:
            self.net = nn.Linear(dim, dim, bias=False)
        if init_fn :
            self.apply(init_fn)

    def forward(self, x, y=None, **kwargs):
        
        if self.linear:
            x = self.net(x)
        else:
            for net in self.net:
                if y is not None:
                    x = adaptive_instance_normalization(x, y)
                x = net(x)

        return x

class SimpleAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., init_fn=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if init_fn:
            self.apply(init_fn)

    def forward(self, tgt, mem=None, attn_mask=None, **kwargs):

        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        if mem is None:
            men = tgt
        q = tgt
        k, v = mem, mem
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, 'b q v -> b 1 q v')

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CusTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout = 0., decoder=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        assert dim % heads == 0
        dim_head = dim//heads

        self.attn_map = list()
        self.cr_attn_map = list()

        
        for _ in range(depth):

            # self_attention_layer = Attention(query_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, )
            self_attention_layer = SimpleAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, )
            # self_attention_layer.processor = CrossAttnStoreProcessor()

            # cross_attention_layer = Attention(query_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, )
            cross_attention_layer = SimpleAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, )
            # cross_attention_layer.processor = CrossAttnStoreProcessor()

            # attention_layer.apply(eye_init)
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                self_attention_layer,
                nn.LayerNorm(dim),
                cross_attention_layer,
                nn.LayerNorm(dim),
                FeedForward(dim, dim, dropout = dropout, ),

            ]))
    
    def get_attn_map(self):
        return self.attn_map

    def forward(self, tgt, mem=None, attn_mask=None, log_attn_map=False):
        self.attn_map = list()
        self.cr_attn_map = list()
        for norm1, attn, norm2, cr_attn, norm3, ff in self.layers:
            tgt = attn(tgt, tgt, attn_mask) + tgt
            tgt = norm1(tgt)
            
            if mem is not None:
                tgt = cr_attn(tgt, mem) + tgt 
                tgt = norm2(tgt)

            tgt = ff(tgt)+tgt
            tgt = norm3(tgt)
            
        return tgt




def draw_random_k(total_sample, num_sample):
    perm = torch.randperm(total_sample)
    idx = perm[:num_sample]
    return idx

 
class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d


        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x






def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
