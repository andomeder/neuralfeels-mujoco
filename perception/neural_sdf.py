import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

    @property
    def output_dim(self) -> int:
        dim = 3 * 2 * self.num_frequencies
        if self.include_input:
            dim += 3
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = []

        if self.include_input:
            encodings.append(x)

        freq_bands = cast(torch.Tensor, self.freq_bands)
        for freq in freq_bands:
            encodings.append(torch.sin(freq * math.pi * x))
            encodings.append(torch.cos(freq * math.pi * x))

        return torch.cat(encodings, dim=-1)


class NeuralSDF(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_frequencies: int = 10,
        skip_connections: list[int] | None = None,
        geometric_init: bool = True,
    ):
        super().__init__()

        self.encoding = PositionalEncoding(num_frequencies=num_frequencies)
        input_dim = self.encoding.output_dim

        if skip_connections is None:
            skip_connections = [4]
        self.skip_connections = skip_connections

        layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            elif i in skip_connections:
                in_features = hidden_dim + input_dim
            else:
                in_features = hidden_dim

            out_features = hidden_dim if i < num_layers - 1 else 1

            layer = nn.Linear(in_features, out_features)

            if geometric_init:
                self._geometric_init(layer, i, num_layers, hidden_dim)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers

    def _geometric_init(
        self,
        layer: nn.Linear,
        layer_idx: int,
        num_layers: int,
        hidden_dim: int,
    ):
        if layer_idx == num_layers - 1:
            nn.init.normal_(layer.weight, mean=0.0, std=0.0001)
            nn.init.constant_(layer.bias, 0.0)
        elif layer_idx == 0:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(hidden_dim))
        else:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(hidden_dim))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = self.encoding(points)
        input_encoding = x

        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input_encoding], dim=-1)

            x = layer(x)

            if i < self.num_layers - 1:
                x = F.softplus(x, beta=100)

        learned = x.squeeze(-1)
        sphere_sdf = torch.norm(points, dim=-1) - 0.05
        return learned + sphere_sdf

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        points = points.requires_grad_(True)
        sdf = self.forward(points)

        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad


def eikonal_loss(gradients: torch.Tensor) -> torch.Tensor:
    grad_norm = torch.linalg.norm(gradients, dim=-1)
    return ((grad_norm - 1.0) ** 2).mean()


def surface_loss(sdf_values: torch.Tensor) -> torch.Tensor:
    return sdf_values.abs().mean()


def free_space_loss(sdf_values: torch.Tensor, target_sdf: torch.Tensor) -> torch.Tensor:
    distance_loss = F.l1_loss(sdf_values, target_sdf)
    negative_penalty = torch.relu(-sdf_values).mean()
    return distance_loss + 10.0 * negative_penalty


def sdf_loss(
    model: NeuralSDF,
    surface_points: torch.Tensor,
    free_points: torch.Tensor,
    free_sdf: torch.Tensor,
    lambda_surface: float = 1.0,
    lambda_free: float = 1.0,
    lambda_eikonal: float = 0.001,
) -> dict[str, torch.Tensor]:
    surface_sdf = model(surface_points)
    loss_surface = surface_loss(surface_sdf)

    surface_grad = model.gradient(surface_points)
    loss_eikonal = eikonal_loss(surface_grad)

    free_sdf_pred = model(free_points)
    loss_free = free_space_loss(free_sdf_pred, free_sdf)

    total_loss = (
        lambda_surface * loss_surface + lambda_free * loss_free + lambda_eikonal * loss_eikonal
    )

    return {
        "total": total_loss,
        "surface": loss_surface,
        "free_space": loss_free,
        "eikonal": loss_eikonal,
    }


def extract_mesh(
    model: NeuralSDF,
    resolution: int = 128,
    bounds: tuple[float, float] = (-0.1, 0.1),
    device: torch.device | str = "cpu",
) -> tuple:
    from skimage import measure

    model.eval()

    x = torch.linspace(bounds[0], bounds[1], resolution)
    y = torch.linspace(bounds[0], bounds[1], resolution)
    z = torch.linspace(bounds[0], bounds[1], resolution)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(device)

    batch_size = 65536
    sdf_values = []

    with torch.no_grad():
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            sdf = model(batch)
            sdf_values.append(sdf.cpu())

    sdf_grid = torch.cat(sdf_values).reshape(resolution, resolution, resolution).numpy()

    try:
        verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=0.0)

        scale = (bounds[1] - bounds[0]) / resolution
        verts = verts * scale + bounds[0]

        return verts, faces, normals
    except ValueError:
        return None, None, None
