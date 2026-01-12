import torch
import torch.nn as nn

from perception.neural_sdf import NeuralSDF, sdf_loss


def test_recovery():
    device = torch.device("cpu")
    model = NeuralSDF(hidden_dim=64, num_layers=4).to(device)

    # Force the model to output negative values initially
    nn.init.constant_(model.layers[-1].bias, -0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    surface_pts = torch.randn(100, 3) * 0.05
    free_pts = torch.randn(100, 3) * 0.1
    # Target SDF is distance to nearest surface point (all positive)
    free_sdf = torch.linalg.norm(free_pts[:, None] - surface_pts[None], dim=-1).min(dim=1)[0]

    print(f"Initial SDF mean (should be negative): {model(free_pts).mean().item():.4f}")

    for i in range(100):
        optimizer.zero_grad()
        losses = sdf_loss(model, surface_pts, free_pts, free_sdf)
        losses["total"].backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            current_mean = model(free_pts).mean().item()
            print(f"Iter {i+1}, Loss: {losses['total'].item():.4f}, SDF mean: {current_mean:.4f}")

    final_mean = model(free_pts).mean().item()
    assert (
        final_mean > 0
    ), f"Model failed to recover from negative initialization. Final mean: {final_mean}"
    print("✓ Model recovered successfully!")


if __name__ == "__main__":
    test_recovery()
