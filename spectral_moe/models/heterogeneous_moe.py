"""Heterogeneous Top-2 mixture-of-experts regressor."""

from __future__ import annotations

import math

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
        raise ImportError("heterogeneous_moe requires PyTorch.")


if nn is not None:

    class MLPExpert(nn.Module):
        """Fully connected expert for resampled spectra and physics features."""


        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden_dim)
            self.block1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.block2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.out = nn.Linear(hidden_dim, out_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.proj(x)
            h = h + self.block1(h)
            h = h + self.block2(h)
            return self.out(h)


    class MultiScaleCNNExpert(nn.Module):


        def __init__(
            self,
            in_dim: int,
            conv_channels: int,
            out_dim: int,
            kernels: list[int] | None = None,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            kernels = kernels or [3, 5, 7]
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, conv_channels, k, padding=k // 2),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                    nn.Conv1d(conv_channels, conv_channels, k, padding=k // 2),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                )
                for k in kernels
            ])
            fused_dim = conv_channels * len(kernels)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, out_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":

            seq = x.unsqueeze(1)
            outs = [self.pool(branch(seq)).squeeze(-1) for branch in self.branches]
            return self.head(torch.cat(outs, dim=-1))

    class RawSpectrumCNNExpert(nn.Module):


        def __init__(self, out_dim: int, spectral_length: int = 2048,
                     channels: int = 32, dropout: float = 0.1) -> None:
            super().__init__()
            self.spectral_length = int(spectral_length)
            c = int(channels)
            self.features = nn.Sequential(
                nn.Conv1d(1, c, kernel_size=9, stride=4, padding=4), nn.GroupNorm(4, c), nn.GELU(),
                nn.Conv1d(c, c * 2, kernel_size=7, stride=4, padding=3), nn.GroupNorm(4, c * 2), nn.GELU(),
                nn.Conv1d(c * 2, c * 2, kernel_size=5, stride=2, padding=2), nn.GroupNorm(4, c * 2), nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Linear(c * 4, c * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(c * 2, out_dim)
            )

        def forward(self, raw_spectrum: "torch.Tensor | None", batch_size: int, device, dtype) -> "torch.Tensor":
            if raw_spectrum is None:
                raw_spectrum = torch.zeros(batch_size, self.spectral_length, device=device, dtype=dtype)
            if raw_spectrum.ndim != 2:
                raise ValueError("raw_spectrum must have shape [B, L]")
            if raw_spectrum.shape[1] != self.spectral_length:
                raw_spectrum = F.interpolate(raw_spectrum.unsqueeze(1), size=self.spectral_length,
                                             mode="linear", align_corners=False).squeeze(1)
            h = self.features(raw_spectrum.unsqueeze(1))
            pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
            return self.head(pooled)


    class PhysicsMappingExpert(nn.Module):


        def __init__(
            self,
            spectrum_dim: int,
            phys_dim: int,
            out_dim: int,
            trough_indices: list[int] | None = None,
            hidden_dim: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.trough_indices = trough_indices or []
            n_sq = len(self.trough_indices)
            in_dim = spectrum_dim + phys_dim + n_sq
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(
            self,
            spectrum_features: "torch.Tensor",
            f_phys: "torch.Tensor",
        ) -> "torch.Tensor":
            if self.trough_indices:
                troughs = f_phys[:, self.trough_indices]
                trough_sq = troughs ** 2
                x = torch.cat([spectrum_features, f_phys, trough_sq], dim=-1)
            else:
                x = torch.cat([spectrum_features, f_phys], dim=-1)
            return self.mlp(x)


    class TransformerExpert(nn.Module):


        def __init__(
            self,
            spectrum_dim: int,
            embed_dim: int = 32,
            n_heads: int = 4,
            n_layers: int = 2,
            out_dim: int = 64,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()

            embed_dim = max(n_heads, (embed_dim // n_heads) * n_heads)
            self.token_emb = nn.Linear(1, embed_dim)
            pos_emb = self._make_sinusoidal_pos(spectrum_dim, embed_dim)
            self.register_buffer("pos_emb", pos_emb)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.out = nn.Linear(embed_dim, out_dim)

        @staticmethod
        def _make_sinusoidal_pos(seq_len: int, dim: int) -> "torch.Tensor":
            pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
            i = torch.arange(0, dim, 2, dtype=torch.float32)
            pe = torch.zeros(seq_len, dim)
            pe[:, 0::2] = torch.sin(pos / (10000 ** (i / dim)))
            if dim % 2 == 1:
                pe[:, 1::2] = torch.cos(pos / (10000 ** (i[:-1] / dim)))
            else:
                pe[:, 1::2] = torch.cos(pos / (10000 ** (i / dim)))
            return pe.unsqueeze(0)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":

            tokens = self.token_emb(x.unsqueeze(-1)) + self.pos_emb
            encoded = self.transformer(tokens)
            pooled = self.norm(encoded.mean(dim=1))
            return self.out(pooled)


    class RawSpectrumMambaExpert(nn.Module):


        def __init__(self, out_dim: int, spectral_length: int = 2048, d_model: int = 64,
                     d_state: int = 64, patch_size: int = 16, dropout: float = 0.1,
                     backend: str = "mamba2") -> None:
            super().__init__()
            if backend != "mamba2":
                raise ValueError("only verified backend is 'mamba2'; Mamba-3 is unavailable on this GPU")
            if spectral_length < patch_size or spectral_length % patch_size:
                raise ValueError("spectral_length must be a positive multiple of patch_size")
            try:
                from mamba_ssm import Mamba2
            except ImportError as exc:
                raise ImportError("RawSpectrumMambaExpert requires mamba-ssm in the training environment") from exc
            self.spectral_length = int(spectral_length)
            self.patch = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
            self.norm_in = nn.LayerNorm(d_model)
            self.forward_block = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            self.backward_block = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            self.norm_out = nn.LayerNorm(d_model)
            self.out = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, out_dim)
            )

        def forward(self, raw_spectrum: "torch.Tensor | None", batch_size: int, device, dtype) -> "torch.Tensor":
            if raw_spectrum is None:
                raw_spectrum = torch.zeros(batch_size, self.spectral_length, device=device, dtype=dtype)
            if raw_spectrum.ndim != 2:
                raise ValueError("raw_spectrum must have shape [B, L]")
            if raw_spectrum.shape[1] != self.spectral_length:
                raw_spectrum = F.interpolate(raw_spectrum.unsqueeze(1), size=self.spectral_length,
                                             mode="linear", align_corners=False).squeeze(1)
            tokens = self.patch(raw_spectrum.unsqueeze(1)).transpose(1, 2)
            tokens = self.norm_in(tokens)
            forward = self.forward_block(tokens)
            backward = torch.flip(self.backward_block(torch.flip(tokens, dims=[1])), dims=[1])
            encoded = self.norm_out(forward + backward)
            pooled = torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)
            return self.out(pooled)


    class HeterogeneousTopKRouter(nn.Module):


        def __init__(self, input_dim: int, num_experts: int = 4, top_k: int = 2, temperature: float = 1.0) -> None:
            super().__init__()
            self.top_k = top_k
            self.num_experts = num_experts
            self.temperature = max(float(temperature), 1e-4)
            self.gate = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, num_experts),
            )

        def forward(
            self, x: "torch.Tensor"
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            logits = self.gate(x)
            weights = torch.softmax(logits / self.temperature, dim=-1)
            top_vals, top_idx = torch.topk(weights, self.top_k, dim=-1)
            sparse = torch.zeros_like(weights).scatter(1, top_idx, top_vals)
            sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return sparse, logits

    class ConditionEncoder(nn.Module):


        def __init__(self, in_dim: int, condition_dim: int = 32, dropout: float = 0.1) -> None:
            super().__init__()
            hidden = condition_dim * 2
            self.encoder = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, condition_dim),
                nn.GELU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.encoder(x)

    class FeatureWiseAffineFiLM(nn.Module):


        def __init__(self, condition_dim: int, hidden_dim: int, scale: float = 0.1) -> None:
            super().__init__()
            self.scale = scale
            self.gamma_proj = nn.Linear(condition_dim, hidden_dim)
            self.beta_proj = nn.Linear(condition_dim, hidden_dim)
            nn.init.zeros_(self.gamma_proj.weight)
            nn.init.zeros_(self.gamma_proj.bias)
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)

        def forward(self, h: "torch.Tensor", cond: "torch.Tensor") -> "torch.Tensor":
            gamma = self.gamma_proj(cond)
            beta = self.beta_proj(cond)
            return h * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * beta


    class QuadraticPhysicsTemperatureHead(nn.Module):


        def __init__(
            self,
            head_hidden_dim: int,
            n_troughs: int,
            dropout: float = 0.1,
            residual_scale_init: float = 0.1,
            global_bias_max: float = 0.0,
        ) -> None:
            super().__init__()
            self.n_troughs = n_troughs
            if n_troughs > 0:
                self.quad_base = nn.Linear(2 * n_troughs, 1)
                residual_in = head_hidden_dim + 2 * n_troughs
            else:
                self.quad_base = None
                residual_in = head_hidden_dim

            self.residual_mlp = nn.Sequential(
                nn.LayerNorm(residual_in),
                nn.Linear(residual_in, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, head_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim // 2, 1),
            )


            init_clamped = max(0.01, min(0.99, residual_scale_init))
            self.residual_logit = nn.Parameter(
                torch.tensor(math.log(init_clamped / (1.0 - init_clamped)))
            )


            self.global_bias_max = float(global_bias_max)
            if self.global_bias_max > 0:
                self.global_bias_raw = nn.Parameter(torch.zeros(1))

        def forward(
            self,
            temp_shared: "torch.Tensor",
            troughs: "torch.Tensor | None",
            trough_sq: "torch.Tensor | None",
        ) -> "torch.Tensor":
            if self.n_troughs > 0 and troughs is not None and trough_sq is not None:
                T_base = self.quad_base(
                    torch.cat([troughs, trough_sq], dim=-1)
                )
                T_residual = self.residual_mlp(torch.cat([temp_shared, troughs, trough_sq], dim=-1))
                out = T_base + torch.sigmoid(self.residual_logit) * T_residual
            else:

                out = self.residual_mlp(temp_shared)
            if self.global_bias_max > 0:
                out = out + torch.tanh(self.global_bias_raw) * self.global_bias_max
            return out

    class LinearPhysicsSalinityHead(nn.Module):


        def __init__(
            self,
            head_hidden_dim: int,
            n_troughs: int,
            dropout: float = 0.1,
            residual_scale_init: float = 0.1,
            deep_residual: bool = False,
            global_bias_max: float = 0.0,
        ) -> None:
            super().__init__()
            self.n_troughs = n_troughs
            if n_troughs > 0:
                self.linear_base = nn.Linear(n_troughs, 1)
                residual_in = head_hidden_dim + n_troughs
            else:
                self.linear_base = None
                residual_in = head_hidden_dim

            if deep_residual:

                self.residual_mlp = nn.Sequential(
                    nn.LayerNorm(residual_in),
                    nn.Linear(residual_in, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, head_hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim // 2, 1),
                )
            else:
                self.residual_mlp = nn.Sequential(
                    nn.LayerNorm(residual_in),
                    nn.Linear(residual_in, head_hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim // 2, 1),
                )

            init_clamped = max(0.01, min(0.99, residual_scale_init))
            self.residual_logit = nn.Parameter(
                torch.tensor(math.log(init_clamped / (1.0 - init_clamped)))
            )

            self.global_bias_max = float(global_bias_max)
            if self.global_bias_max > 0:
                self.global_bias_raw = nn.Parameter(torch.zeros(1))

        def forward(
            self,
            sal_shared: "torch.Tensor",
            troughs: "torch.Tensor | None",
        ) -> "torch.Tensor":
            if self.n_troughs > 0 and troughs is not None:
                S_base = self.linear_base(troughs)
                S_residual = self.residual_mlp(torch.cat([sal_shared, troughs], dim=-1))
                out = S_base + torch.sigmoid(self.residual_logit) * S_residual
            else:
                out = self.residual_mlp(sal_shared)
            if self.global_bias_max > 0:
                out = out + torch.tanh(self.global_bias_raw) * self.global_bias_max
            return out


    class HeterogeneousMoE(nn.Module):


        def __init__(
            self,
            spectrum_dim: int,
            phys_dim: int,
            expert_out_dim: int = 64,
            hidden_dim: int = 128,
            top_k: int = 2,
            trough_indices: list[int] | None = None,
            dropout: float = 0.1,
            head_hidden_dim: int = 64,
            decouple_temperature: bool = True,
            condition_film_cfg: dict | None = None,
            physics_heads_cfg: dict | None = None,
            expert_types: list[str] | None = None,
            use_moe: bool = True,
            mamba_cfg: dict | None = None,
        ) -> None:
            super().__init__()
            self.spectrum_dim = spectrum_dim
            self.phys_dim = phys_dim
            self.trough_indices = list(trough_indices or [])


            self.decouple_temperature = decouple_temperature
            _all_types = ["mlp", "cnn", "physics", "mamba", "transformer"]
            self.use_moe = bool(use_moe)
            if not self.use_moe:
                self.active_expert_types = ["single_shared"]
            elif expert_types is None:


                self.active_expert_types = ["mlp", "cnn", "physics", "transformer"]
            else:
                normalized = [str(t).lower() for t in expert_types]
                self.active_expert_types = [t for t in normalized if t in _all_types]
                if not self.active_expert_types:
                    raise ValueError(f"expert_types must contain at least one of {_all_types}")
            self._n_experts = len(self.active_expert_types)


            in_dim = spectrum_dim + phys_dim
            self.shared_proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )


            if not self.use_moe:
                self.single_shared_expert = MLPExpert(hidden_dim, hidden_dim, expert_out_dim, dropout)
            if "mlp" in self.active_expert_types:
                self.expert_mlp = MLPExpert(hidden_dim, hidden_dim, expert_out_dim, dropout)

            self._cnn_input = "raw"
            if "cnn" in self.active_expert_types:
                cnn_cfg = (mamba_cfg or {}).get("cnn", {})
                self._cnn_input = str(cnn_cfg.get("input", "raw")).lower()
                conv_ch = int(cnn_cfg.get("channels", max(hidden_dim // 4, 16)))
                if self._cnn_input == "raw":
                    self.expert_cnn = RawSpectrumCNNExpert(expert_out_dim, spectral_length=int(cnn_cfg.get("spectral_length", 2048)), channels=conv_ch, dropout=dropout)
                elif self._cnn_input == "latent":
                    self.expert_cnn = MultiScaleCNNExpert(hidden_dim, conv_ch, expert_out_dim, kernels=[3, 5, 7], dropout=dropout)
                else:
                    raise ValueError("mamba.cnn.input must be 'raw' or 'latent'")

            if "physics" in self.active_expert_types:
                self.expert_phys = PhysicsMappingExpert(
                    spectrum_dim, phys_dim, expert_out_dim,
                    trough_indices=self.trough_indices,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )

            if "mamba" in self.active_expert_types:
                mamba_cfg = mamba_cfg or {}
                self.expert_mamba = RawSpectrumMambaExpert(
                    expert_out_dim,
                    spectral_length=int(mamba_cfg.get("spectral_length", 2048)),
                    d_model=int(mamba_cfg.get("d_model", 64)),
                    d_state=int(mamba_cfg.get("d_state", 64)),
                    patch_size=int(mamba_cfg.get("patch_size", 16)),
                    dropout=dropout,
                    backend=str(mamba_cfg.get("backend", "mamba2")),
                )

            if "transformer" in self.active_expert_types:
                transformer_cfg = (mamba_cfg or {}).get("transformer", {})
                self.expert_transformer = TransformerExpert(
                    spectrum_dim=spectrum_dim,
                    embed_dim=int(transformer_cfg.get("embed_dim", 32)),
                    n_heads=int(transformer_cfg.get("n_heads", 4)),
                    n_layers=int(transformer_cfg.get("n_layers", 2)),
                    out_dim=expert_out_dim,
                    dropout=dropout,
                )


            actual_top_k = min(top_k, self._n_experts)
            # The retained main model always uses Top-2 sparse routing.
            self.router = (HeterogeneousTopKRouter(
                hidden_dim, num_experts=self._n_experts, top_k=actual_top_k
            ) if self.use_moe else None)


            film_cfg = condition_film_cfg or {}
            self.use_condition_film = bool(film_cfg.get("enabled", False))
            if self.use_condition_film:
                self._cond_source = str(film_cfg.get("condition_source", "physics"))
                cond_in_dim = phys_dim if self._cond_source == "physics" else (spectrum_dim + phys_dim)
                cond_dim = int(film_cfg.get("condition_dim", 32))
                cond_drop = float(film_cfg.get("dropout", dropout))
                film_scale = float(film_cfg.get("scale", 0.1))
                self.condition_encoder = ConditionEncoder(cond_in_dim, cond_dim, cond_drop)
                self.film = FeatureWiseAffineFiLM(cond_dim, hidden_dim, film_scale)


            self.shared_head = nn.Sequential(
                nn.LayerNorm(expert_out_dim),
                nn.Linear(expert_out_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            n_troughs = len(self.trough_indices)


            phys_head_cfg = physics_heads_cfg or {}
            self._use_physics_heads = (
                bool(phys_head_cfg.get("enabled", False)) and n_troughs > 0
            )

            if self._use_physics_heads:
                temp_res_scale = float(phys_head_cfg.get("temperature_residual_scale_init", 0.1))
                sal_res_scale = float(phys_head_cfg.get("salinity_residual_scale_init", 0.1))

                gb_temp = float(phys_head_cfg.get("global_bias_max_temperature", 0.0))
                gb_sal = float(phys_head_cfg.get("global_bias_max_salinity", 0.0))
                sal_deep = bool(phys_head_cfg.get("salinity_deep_residual", False))

                self.temperature_head = QuadraticPhysicsTemperatureHead(
                    head_hidden_dim=head_hidden_dim,
                    n_troughs=n_troughs,
                    dropout=dropout,
                    residual_scale_init=temp_res_scale,
                    global_bias_max=gb_temp,
                )
                self.salinity_head = LinearPhysicsSalinityHead(
                    head_hidden_dim=head_hidden_dim,
                    n_troughs=n_troughs,
                    dropout=dropout,
                    residual_scale_init=sal_res_scale,
                    deep_residual=sal_deep,
                    global_bias_max=gb_sal,
                )
            else:

                temp_head_in_dim = head_hidden_dim + n_troughs * 2
                self.temperature_head = nn.Sequential(
                    nn.Linear(temp_head_in_dim, head_hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim * 2, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, 1),
                )
                self.salinity_head = nn.Sequential(
                    nn.Linear(head_hidden_dim + n_troughs, head_hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim // 2, 1),
                )

        def forward(
            self,
            spectrum_features: "torch.Tensor",
            f_phys: "torch.Tensor",
            raw_spectrum: "torch.Tensor | None" = None,
        ) -> dict:

            combined = torch.cat([spectrum_features, f_phys], dim=-1)
            h_shared = self.shared_proj(combined)


            if self.use_condition_film:
                cond_input = f_phys if self._cond_source == "physics" else combined
                cond_emb = self.condition_encoder(cond_input)
                h_shared = self.film(h_shared, cond_emb)

            if self.use_moe:
                route_weights, route_logits = self.router(h_shared)
                expert_outputs = []
                if "mlp" in self.active_expert_types:
                    expert_outputs.append(self.expert_mlp(h_shared))
                if "cnn" in self.active_expert_types:
                    if self._cnn_input == "raw":
                        expert_outputs.append(self.expert_cnn(raw_spectrum, spectrum_features.shape[0], spectrum_features.device, spectrum_features.dtype))
                    else:
                        expert_outputs.append(self.expert_cnn(h_shared))
                if "physics" in self.active_expert_types:
                    expert_outputs.append(self.expert_phys(spectrum_features, f_phys))
                if "mamba" in self.active_expert_types:
                    expert_outputs.append(
                        self.expert_mamba(raw_spectrum, spectrum_features.shape[0], spectrum_features.device, spectrum_features.dtype)
                    )
                if "transformer" in self.active_expert_types:
                    expert_outputs.append(self.expert_transformer(spectrum_features))
                stacked = torch.stack(expert_outputs, dim=1)
                mixed = torch.sum(stacked * route_weights.unsqueeze(-1), dim=1)
            else:
                mixed = self.single_shared_expert(h_shared)
                expert_outputs = [mixed]
                route_weights = torch.ones((h_shared.shape[0], 1), device=h_shared.device, dtype=h_shared.dtype)
                route_logits = torch.zeros_like(route_weights)


            troughs = f_phys[:, self.trough_indices] if self.trough_indices else None
            trough_sq = troughs ** 2 if troughs is not None else None


            mixed_for_temp = mixed.detach() if self.decouple_temperature else mixed
            temp_shared = self.shared_head(mixed_for_temp)

            if self._use_physics_heads:

                temperature = self.temperature_head(temp_shared, troughs, trough_sq)
            else:

                temp_parts = [temp_shared]
                if troughs is not None:
                    temp_parts.extend([troughs, trough_sq])
                temperature = self.temperature_head(torch.cat(temp_parts, dim=-1))


            sal_shared = self.shared_head(mixed)

            if self._use_physics_heads:


                salinity = self.salinity_head(sal_shared, troughs)
            else:

                if troughs is not None:
                    sal_input = torch.cat([sal_shared, troughs], dim=-1)
                else:
                    sal_input = sal_shared
                salinity = self.salinity_head(sal_input)

            prediction = torch.cat([temperature, salinity], dim=-1)

            return {
                "prediction": prediction,
                "route_weights": route_weights,
                "route_logits": route_logits,
                "expert_outputs": expert_outputs,
            }


else:

    class HeterogeneousMoE:
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()
