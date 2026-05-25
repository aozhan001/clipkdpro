import importlib.util
import logging
from pathlib import Path


def load_sampler_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "training" / "offline_sampler.py"
    spec = importlib.util.spec_from_file_location("offline_sampler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_metadata(similarities, entropies=None):
    metadata = []
    for index, similarity in enumerate(similarities):
        entry = {
            "sample_id": f"sample::{index}",
            "image_text_similarity": similarity,
            "caption_quality": 0.2 + 0.1 * index,
            "density_score": 0.5 - 0.05 * index,
            "caption_disagreement": 0.05 * index,
        }
        if entropies is not None:
            entry["teacher_entropy"] = entropies[index]
        metadata.append(entry)
    return metadata


def test_random_top_middle_similarity_return_expected_counts():
    module = load_sampler_module()
    metadata = build_metadata([0.1, 0.2, 0.3, 0.4, 0.5])

    random_indices = module.select_sample_indices(metadata, "random", keep_ratio=0.4, seed=7)
    top_indices = module.select_sample_indices(metadata, "top_similarity", keep_ratio=0.4)
    middle_indices = module.select_sample_indices(metadata, "middle_similarity", keep_ratio=0.4)

    assert len(random_indices) == 2
    assert len(top_indices) == 2
    assert len(middle_indices) == 2
    assert top_indices == [4, 3]
    assert middle_indices == [1, 2]


def test_keep_ratio_is_applied():
    module = load_sampler_module()
    metadata = build_metadata([0.1, 0.2, 0.3, 0.4])

    indices = module.select_sample_indices(metadata, "top_similarity", keep_ratio=0.5)

    assert len(indices) == 2
    assert indices == [3, 2]


def test_high_entropy_warns_and_falls_back_to_random(caplog):
    module = load_sampler_module()
    metadata = build_metadata([0.1, 0.2, 0.3, 0.4, 0.5])

    with caplog.at_level(logging.WARNING):
        entropy_indices = module.select_sample_indices(
            metadata,
            "high_entropy",
            keep_ratio=0.4,
            seed=11,
            epoch=2,
        )

    expected_random_indices = module.select_sample_indices(
        metadata,
        "random",
        keep_ratio=0.4,
        seed=11,
        epoch=2,
    )

    assert len(entropy_indices) == 2
    assert entropy_indices == expected_random_indices
    assert "teacher_entropy" in caplog.text
    assert "Falling back to random filtering" in caplog.text


def test_weighted_strategy_prefers_combined_high_value_samples():
    module = load_sampler_module()
    metadata = [
        {
            "sample_id": "sample::0",
            "image_text_similarity": 0.95,
            "teacher_entropy": 0.10,
            "caption_quality": 0.90,
            "density_score": 0.90,
            "caption_disagreement": 0.10,
        },
        {
            "sample_id": "sample::1",
            "image_text_similarity": 0.70,
            "teacher_entropy": 0.95,
            "caption_quality": 0.80,
            "density_score": 0.60,
            "caption_disagreement": 0.95,
        },
        {
            "sample_id": "sample::2",
            "image_text_similarity": 0.40,
            "teacher_entropy": 0.40,
            "caption_quality": 0.30,
            "density_score": 0.30,
            "caption_disagreement": 0.30,
        },
    ]

    args = type(
        "Args",
        (),
        {
            "sample_weight_similarity": 1.0,
            "sample_weight_entropy": 0.6,
            "sample_weight_disagreement": 0.6,
            "sample_weight_density": 0.2,
            "sample_weight_caption_quality": 0.2,
            "sample_weight_difficulty": 0.0,
        },
    )()

    indices = module.select_sample_indices(
        metadata,
        "weighted",
        keep_ratio=2 / 3,
        seed=5,
        epoch=1,
        args=args,
    )

    assert len(indices) == 2
    assert 1 in indices
    assert 0 in indices


def test_high_density_strategy_uses_density_metadata():
    module = load_sampler_module()
    metadata = build_metadata([0.1, 0.2, 0.3, 0.4])

    indices = module.select_sample_indices(metadata, "high_density", keep_ratio=0.5)

    assert indices == [0, 1]
