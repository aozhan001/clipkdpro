import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def load_module(module_name, relative_path):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    def __init__(self):
        self.samples = [
            (torch.tensor([1.0, 2.0]), torch.tensor([10, 11, 0])),
            (torch.tensor([3.0, 4.0]), torch.tensor([12, 0, 0])),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_offline_teacher_cache_wrapper_returns_cached_features(tmp_path):
    module = load_module("offline_teacher_cache", "src/training/offline_teacher_cache.py")

    cache_path = tmp_path / "teacher_cache.pt"
    torch.save(
        {
            "image_features": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float16),
            "text_features": torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float16),
            "candidate_text_features": torch.tensor(
                [
                    [[0.9, 0.1], [0.8, 0.2]],
                    [[0.1, 0.9], [0.2, 0.8]],
                ],
                dtype=torch.float16,
            ),
            "candidate_text_feature_mask": torch.tensor(
                [
                    [True, False],
                    [True, True],
                ]
            ),
            "metadata": [
                {
                    "sample_id": "fake::0",
                    "caption_length": 2,
                    "image_text_similarity": 0.9,
                    "difficulty_score": 0.1,
                    "caption_quality": 0.8,
                    "density_score": 0.7,
                    "teacher_entropy": 0.2,
                    "caption_disagreement": 0.1,
                },
                {
                    "sample_id": "fake::1",
                    "caption_length": 1,
                    "image_text_similarity": 0.7,
                    "difficulty_score": 0.3,
                    "caption_quality": 0.6,
                    "density_score": 0.5,
                    "teacher_entropy": 0.4,
                    "caption_disagreement": 0.2,
                },
            ],
            "config": {"teacher_model": "RN50"},
            "summary": {"cache_format_version": 2},
        },
        cache_path,
    )

    dataset = FakeDataset()
    dataset = module.SampleIDDatasetWrapper(dataset, ["fake::0", "fake::1"])
    wrapped = module.OfflineTeacherCacheDatasetWrapper(dataset, str(cache_path))

    sample = wrapped[0]
    image, text = sample
    assert torch.equal(image, torch.tensor([1.0, 2.0]))
    assert torch.equal(text, torch.tensor([10, 11, 0]))

    cached = wrapped.get_teacher_cache(index=0)
    assert torch.equal(cached["teacher_image_features"], torch.tensor([0.1, 0.2], dtype=torch.float16))
    assert torch.equal(cached["teacher_text_features"], torch.tensor([0.5, 0.6], dtype=torch.float16))
    assert torch.equal(
        cached["candidate_teacher_text_features"],
        torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float16),
    )
    assert torch.equal(cached["candidate_teacher_text_feature_mask"], torch.tensor([True, False]))
    assert cached["caption_length"] == 2
    assert cached["image_text_similarity"] == 0.9
    assert cached["difficulty_score"] == 0.1
    assert cached["caption_quality"] == 0.8
    assert cached["density_score"] == 0.7
    assert cached["teacher_entropy"] == 0.2
    assert cached["caption_disagreement"] == 0.1

    cached_by_sample_id = wrapped.get_teacher_cache(sample_id="fake::1")
    assert torch.equal(cached_by_sample_id["teacher_image_features"], torch.tensor([0.3, 0.4], dtype=torch.float16))
    assert cached_by_sample_id["caption_length"] == 1

    batch = wrapped.collate_fn([wrapped[0], wrapped[1]])
    assert batch.offline_teacher_cache["candidate_teacher_text_features"].shape == (2, 2, 2)
    assert batch.offline_teacher_cache["candidate_teacher_text_feature_mask"].shape == (2, 2)
