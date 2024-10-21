# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dynamic One-For-All (DOFA) models."""

from functools import partial
from typing import Any, Dict

import kornia.augmentation as K
import torch
import yaml
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from pathlib import Path
from timm.models.vision_transformer import Block
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
                    
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """Compute the 1D sine/cosine position embedding.

    Args:
        embed_dim: Output dimension D for each position. Must be even.
        pos: A list of positions to be encoded, of size (M,).

    Returns:
        Position embeddings of size (M, D).

    Raises:
        AssertionError: If *embed_dim* is not even.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    """Dynamic weight generator for DOFA."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Initialize a new TransformerWeightGenerator instance.

        Args:
            input_dim: Input dimensions.
            output_dim: Output dimensions.
            embed_dim: Embedding dimensions.
            num_heads: Number of heads.
            num_layers: Number of layers.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation='gelu',
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Input mini-batch of size (seq_len, batch, input_dim).

        Returns:
            Weight and bias.
        """
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        # Using the last output to generate bias
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class FCResLayer(nn.Module):
    """Fully-connected residual layer."""

    def __init__(self, linear_size: int = 128) -> None:
        """Initialize a new FCResLayer instance.

        Args:
            linear_size: Size of linear layer.
        """
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.

        Returns:
            Output of the model.
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out: Tensor = x + y
        return out


class DOFAEmbedding(nn.Module):
    """Dynamic One-For-All (DOFA) embedding."""

    def __init__(
        self, dynamic_embed_dim: int, kernel_size: int = 3, embed_dim: int = 1024
    ) -> None:
        """Initialize a new DOFAEmbedding instance.

        Args:
            dynamic_embed_dim: Dimensions of dynamic weight generator.
            kernel_size: Kernel size of the depth-wise convolution.
            embed_dim: Embedding dimensions.
        """
        super().__init__()
        self.dynamic_embed_dim = dynamic_embed_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            dynamic_embed_dim, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(dynamic_embed_dim)

        self._init_weights()

    def _init_weight(self, m: object) -> None:
        """Initialize weights of a single layer.

        Args:
            m: A single layer.
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize weights of all layers."""
        self.weight_generator.apply(self._init_weight)
        self.fclayer.apply(self._init_weight)

    def forward(self, x: Tensor, wavelengths: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Return:
            Output mini-batch and wavelengths.
        """
        inplanes = wavelengths.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = position_embedding(self.dynamic_embed_dim, wavelengths * 1000)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)  # 3x3x3

        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([3,0,1,2])
        
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            x, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves

class Encoder(nn.Module):
    """
    Dynamic One-For-All (DOFA) model for segmentation.

    Reference implementation:

    * https://github.com/microsoft/torchgeo/blob/main/torchgeo/models/dofa.py
    * https://github.com/zhu-xlab/DOFA/blob/master/downstream_tasks/segmentation/models/dofa_vit.py
    
    """
    def __init__(self, 
                 img_size: tuple = (224, 224),
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 pre_norm: bool = False,
                 final_norm: bool = False,
                 interpolate_mode: str ='bicubic',
                 norm_layer: type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 out_layers: int | list[int] = -1,
                 ):
        
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / 'dofa.yaml'
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        sensor_name = cfg["sensor"]
        bands = cfg["bands"]
        wavelengths = cfg["wavelengths"][sensor_name]
        model_wavelengths = [wavelengths[band] for band in bands]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.wavelengths = model_wavelengths
        self.pre_norm = pre_norm
        self.final_norm = final_norm
        self.out_layers = out_layers
        self.interpolate_mode = interpolate_mode
        
        if isinstance(out_layers, int):
            if out_layers == -1:
                out_layers= depth - 1
            self.out_layers = [out_layers]
        elif isinstance(out_layers, list) or isinstance(out_layers, tuple):
            self.out_layers = out_layers
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        super().__init__()
        
        self.patch_embed = DOFAEmbedding(dynamic_embed_dim=128, kernel_size=16, embed_dim=self.embed_dim)
        self.num_patches = (self.img_size[0] // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.drop_after_pos = nn.Dropout(p=self.drop_rate)
        self.norm = norm_layer(self.embed_dim)
        
        self.blocks = nn.ModuleList([Block(self.embed_dim, 
                                           self.num_heads, 
                                           self.mlp_ratio, 
                                           self.qkv_bias, 
                                           norm_layer=norm_layer) for i in range(self.depth)])
        
    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                            (pos_h, pos_w),
                                            self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)
        
    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
        
    def forward(self, x: Tensor):
        B = x.shape[0]
        if self.wavelengths is None:
            raise ValueError("Wavelengths must be provided")
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()
        self.waves = wavelist
        
        x, _ = self.patch_embed(x, self.waves)
        hw = self.img_size[0] // self.patch_embed.kernel_size
        hw_shape = (hw, hw)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)
        
        
        if self.pre_norm:
            x = self.norm(x)
        outs = []
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                if self.final_norm:
                    x = self.norm(x)
            if i in self.out_layers:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        
        return outs
    
def dofa_encoder_base(pretrained: bool = True, *args: Any, **kwargs: Any):
    url: str = "https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_base_e120.pth"
    kwargs |= {'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'out_layers': [2, 5, 8, 11]}
    model = Encoder(*args, **kwargs)
    
    if pretrained:
        model_dict = torch.hub.load_state_dict_from_url(url, progress=True)
        del model_dict["mask_token"]
        del model_dict["projector.weight"], model_dict["projector.bias"]
        
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        assert not missing_keys
        assert not unexpected_keys
    
    return model

def dofa_encoder_large(pretrained: bool = True, *args: Any, **kwargs: Any):
    url: str = "https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_large_e100.pth"
    kwargs |= {'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'out_layers': [3, 7, 11, 23]}
    model = Encoder(*args, **kwargs)
    
    if pretrained:
        model_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location='cpu')
        del model_dict["mask_token"]
        del model_dict["projector.weight"], model_dict["projector.bias"]
        
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        assert not missing_keys
        assert not unexpected_keys
    
    return model

if __name__ == '__main__':
    model = dofa_encoder_base()
    batch_size = 6
    img = torch.rand(batch_size, 4, 224, 224)
    out = model(img)
    print(f"Output length: {len(out)}")
    for i in out:
        print(f"Output feature shape: {i.shape}")        