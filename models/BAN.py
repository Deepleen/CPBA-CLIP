import numpy as np
import torch
from torch import nn


class Zero_Parameter(nn.Module):
    """Bidirectional attention with RGB-centered depth fusion.

    The public constructor and ``forward`` return values intentionally match
    the original BAN module.  When depth is available, a scalar gate is
    predicted for each spatial patch and broadcast over channels.  The gate
    is initialized near zero, so the initial fused feature is almost exactly
    the RGB feature instead of an uncontrolled 50/50 mixture.
    """

    def __init__(self, dim_v, dim_t, dim_out, num_heads=8):
        super().__init__()
        if dim_out % num_heads != 0:
            raise ValueError(
                f"dim_out ({dim_out}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out
        self.scale = dim_out ** -0.5

        # Text projections.
        self.q_proj_pre = nn.Conv1d(dim_t, dim_out, 1)
        self.k_proj_pre_t = nn.Conv1d(dim_t, dim_out, 1)
        self.v_proj_pre_t = nn.Conv1d(dim_t, dim_out, 1)
        self.t_res_proj = nn.Conv1d(dim_t, dim_out, 1)

        # Visual projections.
        self.k_proj_pre_1 = nn.Conv1d(dim_v, dim_out, 1)
        self.v_proj_pre_1 = nn.Conv1d(dim_v, dim_out, 1)
        self.q_proj_out = nn.Conv1d(dim_out, dim_out, 1)

        # Output projections.
        self.proj_post_t = nn.Conv1d(dim_out, dim_out, 1)
        self.proj_post_inter = nn.Conv1d(dim_out, dim_out, 1)

        # Depth is already produced in the RGB embedding space.  An identity
        # initialized 1x1 projection keeps that alignment at the start of
        # training while still allowing the optimizer to adapt it.
        self.depth_projection = nn.Conv1d(dim_v, dim_v, 1, bias=True)

        # One logit per patch, not one logit per channel.  Zeroing the final
        # weights and using a negative bias gives a small initial depth weight
        # (sigmoid(-6) ~= 0.0025), i.e. an RGB-centered identity start.
        gate_hidden = max(dim_v // 4, 1)
        self.depth_gate = nn.Sequential(
            nn.Conv1d(dim_v * 2, gate_hidden, 1),
            nn.GELU(),
            nn.Conv1d(gate_hidden, 1, 1),
        )

        self.prompt_temp_l1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.beta_t = 1
        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        # Preserve ordinary Conv1d initialization for attention projections,
        # then explicitly set the two depth-fusion components.
        nn.init.eye_(self.depth_projection.weight.squeeze(-1))
        nn.init.zeros_(self.depth_projection.bias)
        nn.init.zeros_(self.depth_gate[-1].weight)
        nn.init.constant_(self.depth_gate[-1].bias, -6.0)

    @staticmethod
    def _normalize(features):
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _fuse_depth(self, F_s_rgb, F_s_depth):
        if F_s_depth is None:
            return F_s_rgb

        depth = self.depth_projection(
            F_s_depth.permute(0, 2, 1)
        ).permute(0, 2, 1)
        gate_input = torch.cat([F_s_rgb, depth], dim=-1).permute(0, 2, 1)
        depth_gate = torch.sigmoid(self.depth_gate(gate_input)).permute(0, 2, 1)

        # Residual interpolation: gate=0 is exactly RGB, while gate=1 is the
        # projected depth feature.  Broadcasting over the last dimension
        # makes the decision patch-wise but channel-independent.
        fused = F_s_rgb + depth_gate * (depth - F_s_rgb)
        return self._normalize(fused)

    def forward(self, F_t, F_s_rgb, F_s_depth=None):
        B1, N1, _ = F_t.shape
        B2, N2, _ = F_s_rgb.shape
        if B1 != B2:
            raise ValueError(
                f"Text and RGB batch sizes differ: {B1} vs {B2}"
            )
        if F_s_depth is not None and F_s_depth.shape[:2] != (B2, N2):
            raise ValueError(
                "Depth features must have the same [batch, patch] shape as "
                f"RGB features, got {tuple(F_s_depth.shape)} and "
                f"{tuple(F_s_rgb.shape)}"
            )

        F_s = self._fuse_depth(F_s_rgb, F_s_depth)

        # Step 1: visual -> text.
        q_s = self.k_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1)
        q_s = q_s.reshape(B2, N2, self.num_heads, self.head_dim)

        k_t = self.k_proj_pre_t(F_t.permute(0, 2, 1)).permute(0, 2, 1)
        k_t = k_t.reshape(B1, N1, self.num_heads, self.head_dim)

        v_t = self.v_proj_pre_t(F_t.permute(0, 2, 1)).permute(0, 2, 1)
        v_t = v_t.reshape(B1, N1, self.num_heads, self.head_dim)

        attn_s = torch.einsum("bnkc,bmkc->bknm", q_s, k_t) * self.beta_t
        attn_s = attn_s.softmax(dim=-1)
        F_t_inter = torch.einsum("bknm,bmkc->bnkc", attn_s, v_t)
        F_t_inter = F_t_inter.reshape(B2, N2, self.dim_out)

        # Align the visual-token result to the text-token count.
        F_t_inter = F_t_inter.mean(dim=1, keepdim=True).repeat(1, N1, 1)
        F_t_inter = self.proj_post_inter(
            F_t_inter.permute(0, 2, 1)
        ).permute(0, 2, 1)
        F_t_inter = self._normalize(F_t_inter)

        F_t_proj = self.t_res_proj(F_t.permute(0, 2, 1)).permute(0, 2, 1)
        F_t_inter = self._normalize(F_t_proj + F_t_inter)

        # Step 2: text -> visual.
        q_t = self.q_proj_out(F_t_inter.permute(0, 2, 1)).permute(0, 2, 1)
        q_t = q_t.reshape(B1, N1, self.num_heads, self.head_dim)

        k_s = self.k_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1)
        k_s = k_s.reshape(B2, N2, self.num_heads, self.head_dim)

        v_s = self.v_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1)
        v_s = v_s.reshape(B2, N2, self.num_heads, self.head_dim)

        attn_t = torch.einsum("bnkc,bmkc->bknm", q_t, k_s) * self.beta_t
        attn_t = attn_t.softmax(dim=-1)
        F_t_a = torch.einsum("bknm,bmkc->bnkc", attn_t, v_s)
        F_t_a = F_t_a.reshape(B1, N1, self.dim_out)
        F_t_a = self.proj_post_t(F_t_a.permute(0, 2, 1)).permute(0, 2, 1)
        F_t_a = self._normalize(F_t_inter + F_t_a)

        return F_t_a, F_s
