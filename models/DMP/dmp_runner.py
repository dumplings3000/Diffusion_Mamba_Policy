from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.DMP.model import DMP

class DMPRunner(
        nn.Module,
        CompatiblePyTorchModelHubMixin
    ):
    def __init__(self, *, action_dim, pred_horizon, config,
                 img_token_dim, state_token_dim, pcd_token_dim,
                 img_cond_len, pcd_cond_len, pcd_pos_embed_config=None,
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(DMPRunner, self).__init__()
        hidden_size = config['DMP']['hidden_size']
        self.model = DMP(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            hidden_size=hidden_size,
            img_token_dim=img_token_dim,
            state_token_dim=state_token_dim,
            pcd_token_dim=pcd_token_dim,
            img_cond_len=img_cond_len,
            pcd_cond_len=pcd_cond_len,
            pcd_pos_embed_config=pcd_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype
        )