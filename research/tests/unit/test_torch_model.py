from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eyegaze.adapters.torch_model import EyeGazeNet, count_parameters, euclidean_loss  # noqa: E402
from eyegaze.config import GeometryFeatures, ModelConfig  # noqa: E402

pytestmark = pytest.mark.requires_torch


def _inputs(batch: int = 2, eye: int = 64, face: int = 64) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.rand(batch, 3, eye, eye, generator=g),
        torch.rand(batch, 3, eye, eye, generator=g),
        torch.rand(batch, 3, face, face, generator=g),
        torch.rand(batch, 17, generator=g),
    )


def test_output_shapes() -> None:
    model = EyeGazeNet(ModelConfig(embedding_dim=32)).eval()
    gaze, emb = model(*_inputs())
    assert gaze.shape == (2, 2)
    assert emb.shape == (2, 32)


@pytest.mark.parametrize("eye_size", [64, 96, 128])
def test_accepts_ablation_crop_sizes(eye_size: int) -> None:
    gaze, _ = EyeGazeNet(ModelConfig()).eval()(*_inputs(eye=eye_size))
    assert gaze.shape == (2, 2)


def test_disabled_face_tower_ignores_face_input() -> None:
    model = EyeGazeNet(ModelConfig(use_face=False)).eval()
    r, le, f, g = _inputs()
    a, _ = model(r, le, f, g)
    b, _ = model(r, le, torch.zeros_like(f), g)
    torch.testing.assert_close(a, b)


def test_geometry_none_ignores_geometry_and_no_pose_ignores_pose() -> None:
    r, le, f, g = _inputs()
    none_model = EyeGazeNet(ModelConfig(geometry=GeometryFeatures.NONE)).eval()
    torch.testing.assert_close(none_model(r, le, f, g)[0], none_model(r, le, f, g * 7)[0])

    no_pose = EyeGazeNet(ModelConfig(geometry=GeometryFeatures.NO_POSE)).eval()
    g2 = g.clone()
    g2[:, 12:15] += 1.0
    torch.testing.assert_close(no_pose(r, le, f, g)[0], no_pose(r, le, f, g2)[0])
    g3 = g.clone()
    g3[:, 0] += 1.0
    assert not torch.allclose(no_pose(r, le, f, g)[0], no_pose(r, le, f, g3)[0])


def test_geometry_stats_are_validated_and_clamped() -> None:
    model = EyeGazeNet(ModelConfig())
    model.set_geometry_stats(torch.zeros(17), torch.zeros(17))
    assert float(model.geometry_std.min()) > 0
    with pytest.raises(ValueError, match="shape"):
        model.set_geometry_stats(torch.zeros(3), torch.ones(3))


def test_euclidean_loss_value_and_finite_gradient_at_zero() -> None:
    pred = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    # Mean of 5 and ~0 (sqrt(eps) = 1e-3 keeps the zero-error row differentiable).
    assert float(euclidean_loss(pred, torch.zeros(2, 2))) == pytest.approx(2.5, abs=1e-3)
    at_target = torch.zeros(1, 2, requires_grad=True)
    euclidean_loss(at_target, torch.zeros(1, 2)).backward()
    assert at_target.grad is not None
    assert bool(torch.isfinite(at_target.grad).all())


def test_backward_reaches_every_trainable_parameter() -> None:
    model = EyeGazeNet(ModelConfig()).train()
    gaze, _ = model(*_inputs(batch=4))
    euclidean_loss(gaze, torch.zeros_like(gaze)).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_model_is_mobile_sized() -> None:
    assert count_parameters(EyeGazeNet(ModelConfig())) < 2_000_000


def test_can_overfit_a_tiny_batch() -> None:
    """Sanity check of the whole forward/backward path: 8 samples must be memorized."""
    torch.manual_seed(0)
    model = EyeGazeNet(ModelConfig(width=0.5, dropout=0.0)).train()
    inputs = _inputs(batch=8)
    target = torch.randn(8, 2) * 3
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    with torch.no_grad():
        initial = float(euclidean_loss(model(*inputs)[0], target))
    for _ in range(150):
        optimizer.zero_grad()
        loss = euclidean_loss(model(*inputs)[0], target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = float(euclidean_loss(model(*inputs)[0], target))
    assert final < 0.3 * initial
