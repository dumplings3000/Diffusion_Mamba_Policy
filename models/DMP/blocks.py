import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import _Final

from timm.models.vision_transformer import Mlp, RmsNorm, Attention, use_fused_attn

import math
import numpy as np
from collections import OrderedDict

class DMPBlock(nn.Module):
    """
    Diffusion Mamba Policy (DMP) Block
    """
    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
    
    def forward(self, x, conds):
        origin_x = x
        x = self.norm1(x)
        x = self.attent
