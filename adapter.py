from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("adapter requires PyTorch.")


if nn is not None:

    class AdapterLinear(nn.Module):


        def __init__(
            self,
            linear: "nn.Linear",
            bottleneck_dim: int = 16,
            dropout: float = 0.0,
            scale: float = 1.0,
        ) -> None:
            super().__init__()
            if bottleneck_dim <= 0:
                raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")

            self.in_features = linear.in_features
            self.out_features = linear.out_features
            self.bottleneck_dim = bottleneck_dim
            self.scale = scale
            device = linear.weight.device
            dtype = linear.weight.dtype


            self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
            if linear.bias is not None:
                self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
            else:
                self.bias = None


            self.adapter_down = nn.Linear(self.out_features, bottleneck_dim, device=device, dtype=dtype)
            self.adapter_up = nn.Linear(bottleneck_dim, self.out_features, device=device, dtype=dtype)
            self.adapter_act = nn.GELU()
            self.adapter_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()


            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = F.linear(x, self.weight, self.bias)
            a = self.adapter_up(self.adapter_dropout(self.adapter_act(self.adapter_down(h))))
            return h + self.scale * a

        @property
        def trainable_param_count(self) -> int:
            return int(
                self.adapter_down.weight.numel() + self.adapter_down.bias.numel()
                + self.adapter_up.weight.numel() + self.adapter_up.bias.numel()
            )

        def extra_repr(self) -> str:
            return (
                f"in={self.in_features}, out={self.out_features}, "
                f"bottleneck={self.bottleneck_dim}, scale={self.scale:.2f}"
            )


    def apply_adapter_to_model(
        model: "nn.Module",
        bottleneck_dim: int = 16,
        dropout: float = 0.0,
        scale: float = 1.0,
        target_modules: list[str] | None = None,
        exclude_modules: list[str] | None = None,
        verbose: bool = True,
    ) -> "nn.Module":


        target_modules = target_modules or []
        exclude_modules = exclude_modules or []
        replaced = 0
        module_dict = dict(model.named_modules())

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if target_modules and not any(s in name for s in target_modules):
                continue
            if any(s in name for s in exclude_modules):
                continue

            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = module_dict[parent_name]
            else:
                parent_name, child_name = "", name
                parent = model

            adapter_layer = AdapterLinear(
                module, bottleneck_dim=bottleneck_dim, dropout=dropout, scale=scale
            )
            setattr(parent, child_name, adapter_layer)
            if verbose:
                print(
                    f"  Adapter ← {name}"
                    f"  [{module.in_features}→{module.out_features}]"
                    f"  r={bottleneck_dim}"
                )
            replaced += 1

        if verbose and replaced > 0:
            stats = count_parameters(model)
            pct = 100.0 * stats["adapter"] / max(stats["total"], 1)
            print(
                f"\n[Adapter] {replaced} 层已包裹 | "
                f"可训练={stats['adapter']:,} / 总参数={stats['total']:,} ({pct:.2f}%)"
            )
        elif verbose:
            print("[Adapter] 警告：没有任何层被包裹，请检查 target_modules 参数。")

        return model

    def freeze_non_adapter(model: "nn.Module") -> None:

        for name, param in model.named_parameters():
            if "adapter_down" in name or "adapter_up" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def get_adapter_parameters(model: "nn.Module") -> list["nn.Parameter"]:

        return [
            p for name, p in model.named_parameters()
            if ("adapter_down" in name or "adapter_up" in name) and p.requires_grad
        ]

    def count_parameters(model: "nn.Module") -> dict[str, int]:

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adapter = sum(
            p.numel() for name, p in model.named_parameters()
            if "adapter_down" in name or "adapter_up" in name
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "adapter": adapter,
        }


else:

    class AdapterLinear:
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    def apply_adapter_to_model(*args, **kwargs):
        _require_torch()

    def freeze_non_adapter(*args, **kwargs):
        _require_torch()

    def get_adapter_parameters(*args, **kwargs):
        _require_torch()

    def count_parameters(*args, **kwargs):
        _require_torch()
