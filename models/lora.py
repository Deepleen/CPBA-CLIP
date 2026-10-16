import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 4,
        alpha: int = 1,
        use_rslora: bool = False,
        use_dora: bool = False,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")
        if use_rslora and use_dora:
            raise ValueError("RSLoRA and DoRA cannot be enabled at the same time.")

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.use_rslora = use_rslora
        self.use_dora = use_dora

        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(self.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, self.out_features, bias=False)
        self.scale = alpha / math.sqrt(r) if use_rslora else alpha / r

        if use_dora:
            weight_norm = self.original_linear.weight.detach().norm(dim=1, keepdim=True)
            self.lora_magnitude = nn.Parameter(weight_norm)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        original_out = self.original_linear(x)
        lora_weight = self.lora_b.weight @ self.lora_a.weight

        if self.use_dora:
            adapted_weight = self.original_linear.weight + self.scale * lora_weight
            adapted_direction = adapted_weight / (adapted_weight.norm(dim=1, keepdim=True) + 1e-12)
            dora_weight = self.lora_magnitude * adapted_direction
            dora_out = torch.matmul(x, dora_weight.transpose(0, 1))
            if self.original_linear.bias is not None:
                dora_out = dora_out + self.original_linear.bias
            return dora_out

        lora_out = self.lora_b(self.lora_a(x)) * self.scale
        return original_out + lora_out


def inject_lora_to_depth_visual(
    model,
    r=4,
    alpha=1,
    target_blocks=[-1, -2, -3, -4],
    use_rslora=False,
    use_dora=False,
):
    """
    Inject LoRA into the specified transformer blocks of the depth_visual branch.
    Target blocks are indices (can be negative).

    Args:
        model: The CLIP model containing depth_visual
        r: LoRA rank
        alpha: LoRA scaling factor
        target_blocks: List of block indices to inject LoRA into
        use_rslora: Whether to use RSLoRA scaling
        use_dora: Whether to use DoRA reparameterization
    """
    if use_rslora and use_dora:
        raise ValueError("Only one LoRA variant can be enabled at a time.")

    if not hasattr(model, 'depth_visual'):
        print("Warning: Model has no depth_visual attribute. LoRA injection skipped.")
        return model

    blocks = model.depth_visual.transformer.resblocks
    num_blocks = len(blocks)

    target_indices = set()
    for i in target_blocks:
        if i < 0:
            i = num_blocks + i
        if 0 <= i < num_blocks:
            target_indices.add(i)

    variant_name = "DoRA" if use_dora else "RSLoRA" if use_rslora else "LoRA"
    print(f"Injecting {variant_name} (r={r}, alpha={alpha}) into depth_visual blocks: {sorted(list(target_indices))}")

    for idx in target_indices:
        block = blocks[idx]

        if hasattr(block.attn, 'out_proj') and isinstance(block.attn.out_proj, nn.Linear):
            if not isinstance(block.attn.out_proj, LoRALinear):
                block.attn.out_proj = LoRALinear(
                    block.attn.out_proj,
                    r=r,
                    alpha=alpha,
                    use_rslora=use_rslora,
                    use_dora=use_dora,
                )
    return model


def mark_lora_as_trainable(model):
    """
    Helper function to ensure LoRA parameters are trainable while others are frozen
    (except those already trainable).
    """
    trainable_params = []
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
            trainable_params.append(n)

    print(f"Enabled gradients for {len(trainable_params)} LoRA parameter tensors.")
    return trainable_params
