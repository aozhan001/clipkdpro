import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def load_loss_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "open_clip" / "loss.py"
    spec = importlib.util.spec_from_file_location("open_clip_loss", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_offline_kd_loss_mse_and_cosine():
    module = load_loss_module()

    student = torch.randn(4, 8)
    teacher = torch.randn(4, 8)

    mse_loss = module.compute_offline_feature_kd_loss(student, teacher, loss_type="mse")
    cosine_loss = module.compute_offline_feature_kd_loss(student, teacher, loss_type="cosine")

    assert mse_loss.ndim == 0
    assert cosine_loss.ndim == 0
    assert torch.isfinite(mse_loss)
    assert torch.isfinite(cosine_loss)
    assert mse_loss.item() >= 0
    assert cosine_loss.item() >= 0


def test_candidate_teacher_text_aggregation_returns_valid_features():
    module = load_loss_module()

    student = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    candidates = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.8, 0.2]],
        ]
    )
    mask = torch.tensor([[True, True], [True, False]])

    aggregated_mean = module.aggregate_candidate_teacher_text_features(
        student,
        candidates,
        mask,
        strategy="mean",
    )
    aggregated_top1 = module.aggregate_candidate_teacher_text_features(
        student,
        candidates,
        mask,
        strategy="top1",
    )

    assert aggregated_mean.shape == student.shape
    assert aggregated_top1.shape == student.shape
    assert torch.isfinite(aggregated_mean).all()
    assert torch.isfinite(aggregated_top1).all()
