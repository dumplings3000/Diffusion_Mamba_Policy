import torch
import torch.nn as nn

from models.DMP.blocks import DMPBlock, TimestepEmbedder, FinalLayer

class DMP(nn.Module):
    def __init__(self,
                 output_dim = 128,
                 horizon = 32,
                 hidden_size =1152,
                 depth = 28,
                 num_heads = 16,
                 img_cond_len=4096,
                 pcd_cond_len=4096,
                 img_pos_embed_config=None,
                 pcd_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.img_cond_len = img_cond_len
        self.pcd_cond_len = pcd_cond_len
        self.img_pos_embed_config = img_pos_embed_config
        self.pcd_pos_embed_config = pcd_pos_embed_config
        self.dtype = dtype  

        # Embed
        self.t_embed = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embed = TimestepEmbedder(hidden_size, dtype=dtype)

        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon+3, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))
        self.pcd_cond_pos_embed = nn.Parameter(torch.zeros(1, pcd_cond_len, hidden_size))

        # Blocks
        self.blocks = nn.ModuleList([
            DMPBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, output_dim)

        self.initialize_weights(self)
    
    def forward(self, x, freq, t, img_cond, pcd_cond):
        """
        forward pass of DMP model

        B: batch size
        T: time sequence length
        D: Dimension of hidden state

        x: (B, T, D), state + action token sequence
        freq: (B,), a scalar indicating control frequency.
        t: (B,) or (1,), diffusion timesteps.
        img_cond: (B, L_img, D) or None, image condition tokens (fixed length),
        pcd_cond: (B, L_pcd, D) or None, point cloud condition tokens (fixed length)
        """
        t = self.t_embed(t).unsqueeze(1)  # (B, D) -> (B, 1, D)
        freq = self.freq_embed(freq).unsqueeze(1)  # (B, D) -> (B, 1, D)
        
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        
        x = torch.cat([t, freq, x], dim=1)  # (B, T+2, D)

        # Positional Embedding
        x = x + self.x_pos_embed  # (B, T+2, D)
        img_cond = img_cond + self.img_cond_pos_embed # (B, L_img, D)
        pcd_cond = pcd_cond + self.pcd_cond_pos_embed # (B, L_pcd, D)
        
        # forward pass
        conds = [img_cond, pcd_cond]

        for block in self.blocks:
            x = block(x, conds)  # (B, T+2, D)

        x = self.final_layer(x)  # (B, T+2, D) -> (B, T+2, output_dim)
        x = x[:, -self.horizon:]  # (B, T+2, output_dim) -> (B, horizon, output_dim)
        
        return x
    
        
