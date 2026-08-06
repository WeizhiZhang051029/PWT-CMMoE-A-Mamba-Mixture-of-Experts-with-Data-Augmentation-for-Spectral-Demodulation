"""Conditional WGAN-GP components used by the training pipeline."""

from __future__ import annotations


try:

    import torch

    from torch import nn

except ImportError:

    torch = None

    nn = None


from spectral_moe.models.cnn_blocks import ConvBNAct, require_torch


if nn is not None:


    class CriticConvAct(nn.Module):


        def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int):

            super().__init__()

            padding = (kernel_size - 1) // 2

            self.net = nn.Sequential(

                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),

                nn.LeakyReLU(0.2, inplace=True),

            )


        def forward(self, x):

            return self.net(x)


    class ConditionalGenerator(nn.Module):

        def __init__(self, latent_dim: int, condition_dim: int, output_length: int, base_channels: int = 64):

            super().__init__()

            self.output_length = output_length

            self.coarse_length = 512

            self.fc = nn.Sequential(

                nn.Linear(latent_dim + condition_dim, base_channels * self.coarse_length),

                nn.GELU(),

            )

            self.refine = nn.Sequential(

                ConvBNAct(base_channels, base_channels, kernel_size=7),

                nn.Conv1d(base_channels, 1, kernel_size=7, padding=3),

            )


        def forward(self, z, condition):

            x = torch.cat([z, condition], dim=-1)

            x = self.fc(x).reshape(x.shape[0], -1, self.coarse_length)

            x = torch.nn.functional.interpolate(x, size=self.output_length, mode="linear", align_corners=False)

            return self.refine(x)


    class ConditionalVectorGenerator(nn.Module):


        def __init__(self, latent_dim: int, condition_dim: int, output_dim: int, hidden_dim: int = 256):

            super().__init__()

            self.net = nn.Sequential(

                nn.Linear(latent_dim + condition_dim, hidden_dim), nn.GELU(),

                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),

                nn.Linear(hidden_dim, output_dim),

            )


        def forward(self, z, condition):

            return self.net(torch.cat([z, condition], dim=-1))


    class ConditionalCritic(nn.Module):

        def __init__(self, condition_dim: int, input_length: int, base_channels: int = 32):

            super().__init__()

            self.encoder = nn.Sequential(

                CriticConvAct(1, base_channels, kernel_size=9, stride=4),

                CriticConvAct(base_channels, base_channels * 2, kernel_size=7, stride=4),

                CriticConvAct(base_channels * 2, base_channels * 4, kernel_size=5, stride=4),

                nn.AdaptiveAvgPool1d(1),

            )

            self.head = nn.Sequential(

                nn.Linear(base_channels * 4 + condition_dim, base_channels * 4),

                nn.GELU(),

                nn.Linear(base_channels * 4, 1),

            )


        def forward(self, spectrum, condition):

            features = self.encoder(spectrum).squeeze(-1)

            return self.head(torch.cat([features, condition], dim=-1))


    class ConditionalVectorCritic(nn.Module):


        def __init__(self, condition_dim: int, input_dim: int, hidden_dim: int = 256):

            super().__init__()

            self.net = nn.Sequential(

                nn.Linear(input_dim + condition_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(hidden_dim, 1),

            )


        def forward(self, spectrum, condition):

            return self.net(torch.cat([spectrum, condition], dim=-1))


    def gradient_penalty(critic, real, fake, condition):

        batch = real.shape[0]

        eps = torch.rand((batch,) + (1,) * (real.ndim - 1), device=real.device)

        mixed = eps * real + (1 - eps) * fake

        mixed.requires_grad_(True)

        score = critic(mixed, condition)

        grad = torch.autograd.grad(

            outputs=score,

            inputs=mixed,

            grad_outputs=torch.ones_like(score),

            create_graph=True,

            retain_graph=True,

            only_inputs=True,

        )[0]

        return ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()


    def zero_centered_gradient_penalty(critic, samples, condition):

        samples = samples.requires_grad_(True)

        score = critic(samples, condition)

        grad = torch.autograd.grad(

            outputs=score,

            inputs=samples,

            grad_outputs=torch.ones_like(score),

            create_graph=True,

            retain_graph=True,

            only_inputs=True,

        )[0]

        return grad.flatten(1).square().sum(dim=1).mean()


    def r3gan_critic_loss(critic, real, fake, condition, r1_weight=0.0, r2_weight=0.0):

        real_score = critic(real, condition)

        fake_score = critic(fake, condition)

        loss = torch.nn.functional.softplus(-(real_score - fake_score)).mean()

        if r1_weight > 0:

            loss = loss + 0.5 * r1_weight * zero_centered_gradient_penalty(critic, real, condition)

        if r2_weight > 0:

            loss = loss + 0.5 * r2_weight * zero_centered_gradient_penalty(critic, fake, condition)

        return loss


    def r3gan_generator_loss(critic, real, fake, condition):

        real_score = critic(real, condition).detach()

        fake_score = critic(fake, condition)

        return torch.nn.functional.softplus(real_score - fake_score).mean()


else:


    class ConditionalGenerator:

        def __init__(self, *args, **kwargs):

            require_torch()


    class ConditionalVectorGenerator:

        def __init__(self, *args, **kwargs):

            require_torch()


    class ConditionalCritic:

        def __init__(self, *args, **kwargs):

            require_torch()


    class ConditionalVectorCritic:

        def __init__(self, *args, **kwargs):

            require_torch()


    def gradient_penalty(*args, **kwargs):

        require_torch()


    def zero_centered_gradient_penalty(*args, **kwargs):

        require_torch()


    def r3gan_critic_loss(*args, **kwargs):

        require_torch()


    def r3gan_generator_loss(*args, **kwargs):

        require_torch()
