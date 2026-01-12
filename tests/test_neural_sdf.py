import torch


def test_neural_sdf_creation():
    from perception.neural_sdf import NeuralSDF

    model = NeuralSDF(hidden_dim=256, num_layers=8)
    num_params = sum(p.numel() for p in model.parameters())

    assert num_params > 400000


def test_neural_sdf_forward():
    from perception.neural_sdf import NeuralSDF

    model = NeuralSDF(hidden_dim=64, num_layers=4)
    points = torch.randn(100, 3)
    sdf = model(points)

    assert sdf.shape == (100,)


def test_neural_sdf_gradient():
    from perception.neural_sdf import NeuralSDF

    model = NeuralSDF(hidden_dim=64, num_layers=4)
    points = torch.randn(50, 3)
    grad = model.gradient(points)

    assert grad.shape == (50, 3)


def test_positional_encoding():
    from perception.neural_sdf import PositionalEncoding

    enc = PositionalEncoding(num_frequencies=10, include_input=True)
    x = torch.randn(32, 3)
    encoded = enc(x)

    assert encoded.shape == (32, enc.output_dim)
    assert enc.output_dim == 3 + 3 * 2 * 10


def test_sdf_loss():
    from perception.neural_sdf import NeuralSDF, sdf_loss

    model = NeuralSDF(hidden_dim=64, num_layers=4)
    surface_pts = torch.randn(20, 3) * 0.05
    free_pts = torch.randn(20, 3) * 0.1
    free_sdf = torch.ones(20) * 0.05

    losses = sdf_loss(model, surface_pts, free_pts, free_sdf)

    assert "total" in losses
    assert "surface" in losses
    assert "eikonal" in losses
    assert "free_space" in losses


def test_free_space_loss_provides_gradient_when_negative():
    from perception.neural_sdf import free_space_loss

    sdf_pred = torch.full((32,), -0.1, requires_grad=True)
    target = torch.full((32,), 0.05)

    loss = free_space_loss(sdf_pred, target)
    loss.backward()

    assert sdf_pred.grad is not None
    assert torch.isfinite(sdf_pred.grad).all()
    assert (sdf_pred.grad.abs() > 0).any()


if __name__ == "__main__":
    test_neural_sdf_creation()
    print("✓ test_neural_sdf_creation passed")

    test_neural_sdf_forward()
    print("✓ test_neural_sdf_forward passed")

    test_neural_sdf_gradient()
    print("✓ test_neural_sdf_gradient passed")

    test_positional_encoding()
    print("✓ test_positional_encoding passed")

    test_sdf_loss()
    print("✓ test_sdf_loss passed")

    print("\nAll Neural SDF tests passed!")
