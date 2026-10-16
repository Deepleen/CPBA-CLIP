"""CPG-V3: static multi-convolutional context prompting.

This version keeps only the operations requested for the simplified ablation:
parallel convolutions on the raw CLIP embedding, input-independent fusion,
and an additive prompt residual.  It intentionally has no normalization,
activation, dynamic router/softmax, dense semantic branch, or residual gate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class StaticMultiScalePromptAdapter(nn.Module):
    """Map ``[B, D]`` features to ``[B, P, D]`` with static fusion."""

    def __init__(self, width: int, num_prompts: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if num_prompts <= 0:
            raise ValueError(f"num_prompts must be positive, got {num_prompts}")

        self.width = width
        self.num_prompts = num_prompts
        self.kernel_sizes = (1, 3, 5)

        # Each branch sees the raw embedding directly and produces one
        # residual for every virtual prompt.
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=num_prompts,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True,
                )
                for kernel_size in self.kernel_sizes
            ]
        )

        # Learned but input-independent 1x1 fusion.  Unlike a router, this
        # layer uses the same fusion weights for every image.
        self.static_fusion = nn.Conv1d(
            in_channels=len(self.kernel_sizes) * num_prompts,
            out_channels=num_prompts,
            kernel_size=1,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for branch in self.branches:
            nn.init.kaiming_uniform_(branch.weight, a=5**0.5)
            nn.init.zeros_(branch.bias)

        # Start from an equal, prompt-aligned average of the three scales.
        # The fusion remains fully learnable after initialization.
        nn.init.zeros_(self.static_fusion.weight)
        nn.init.zeros_(self.static_fusion.bias)
        for prompt_index in range(self.num_prompts):
            for branch_index in range(len(self.kernel_sizes)):
                channel_index = branch_index * self.num_prompts + prompt_index
                self.static_fusion.weight.data[
                    prompt_index, channel_index, 0
                ] = 1.0 / len(self.kernel_sizes)

    def _validate_feature(self, global_feature: torch.Tensor) -> torch.Tensor:
        if global_feature.ndim == 3:
            if global_feature.shape[1] != 1:
                raise ValueError(
                    "global_feature with 3 dimensions must have shape [B, 1, D], "
                    f"got {tuple(global_feature.shape)}"
                )
            global_feature = global_feature[:, 0, :]
        if global_feature.ndim != 2 or global_feature.shape[-1] != self.width:
            raise ValueError(
                "global_feature must have shape [B, D] or [B, 1, D], "
                f"got {tuple(global_feature.shape)} with D={self.width}"
            )
        return global_feature

    def forward(self, global_feature: torch.Tensor) -> torch.Tensor:
        global_feature = self._validate_feature(global_feature)
        raw_feature = global_feature.unsqueeze(1)
        branch_outputs = [branch(raw_feature) for branch in self.branches]
        fused_input = torch.cat(branch_outputs, dim=1)
        return self.static_fusion(fused_input)


class Context_Prompting(nn.Module):
    """Shared-prompt CPG-V3 with additive static multi-scale residuals."""

    def __init__(self, model_config: dict, cla_len: int) -> None:
        super().__init__()
        width = model_config["text_cfg"]["width"]
        if width != model_config["embed_dim"]:
            raise ValueError(
                "Context_Prompting requires text width == embedding width; "
                f"got text width {width} and embed_dim {model_config['embed_dim']}"
            )

        self.cla_len = cla_len
        self.prompt_query_global = nn.Parameter(torch.empty(1, cla_len, width))
        self.prompt_query_regional = nn.Parameter(torch.empty(1, cla_len, width))
        self.prompt_query = nn.Parameter(torch.empty(1, cla_len, width))
        # Keep this attribute name so existing optimizer/checkpoint code can
        # use the module after changing only the import statement.
        self.prompt_linear1 = StaticMultiScalePromptAdapter(
            width=width,
            num_prompts=cla_len,
        )
        self.prompt_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # nn.init.trunc_normal_(self.prompt_query)
        # Start both branches from exactly the same learned-prompt distribution.
        nn.init.trunc_normal_(self.prompt_query_global)
        with torch.no_grad():
            self.prompt_query_regional.copy_(self.prompt_query_global)

    def before_extract_feat(
        self,
        x: torch.Tensor | None,
        img_feature: torch.Tensor,
        use_global: bool = True,
    ) -> torch.Tensor:
        del x  # Existing callers pass patch tokens as the first argument.
        batch_size = img_feature.shape[0]
        prompt_query = self.prompt_query_regional.expand(batch_size, -1, -1)
        if not use_global:
            return self.prompt_query_regional.expand(batch_size, -1, -1)

        prompt_query = self.prompt_query_global.expand(batch_size, -1, -1)
        global_delta = self.prompt_linear1(img_feature)
        return prompt_query + global_delta.to(dtype=prompt_query.dtype)