import importlib.util
from pathlib import Path

import pytest


def load_cache_teacher_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "training" / "cache_teacher_features.py"
    spec = importlib.util.spec_from_file_location("cache_teacher_features", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cache_teacher_features_parse():
    module = load_cache_teacher_module()
    args = module.parse_args(
        [
            "--output-cache",
            "tmp/teacher_cache.pt",
            "--teacher-model",
            "RN50",
            "--dataset-type",
            "synthetic",
            "--train-num-samples",
            "8",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--csv-caption-candidates-keys",
            "synthetic_caption,llm_caption",
            "--csv-metadata-keys",
            "caption_quality_score,teacher_disagreement",
            "--cache-density-k",
            "4",
        ]
    )

    assert args.output_cache == "tmp/teacher_cache.pt"
    assert args.teacher_model == "RN50"
    assert args.dataset_type == "synthetic"
    assert args.train_num_samples == 8
    assert args.csv_caption_candidates_keys == "synthetic_caption,llm_caption"
    assert args.csv_metadata_keys == "caption_quality_score,teacher_disagreement"
    assert args.cache_density_k == 4


def test_cache_teacher_helper_functions_cover_candidate_metadata():
    module = load_cache_teacher_module()

    assert module.get_csv_candidate_caption_keys(
        type(
            "Args",
            (),
            {
                "csv_caption_key": "title",
                "csv_caption_candidates_keys": "title,synthetic_caption,llm_caption",
                "csv_synthetic_caption_keys": "",
            },
        )()
    ) == ["synthetic_caption", "llm_caption"]
    assert module.compute_caption_quality_score("a clean detailed caption of a dog") > 0
    assert module.compute_similarity_entropy(0.9, [0.7, 0.4]) is not None


def test_cache_teacher_csv_dataset_with_coco_in_filename_is_not_treated_as_retrieval(tmp_path):
    module = load_cache_teacher_module()

    try:
        from training.data import is_retrieval_dataset_path
    except ModuleNotFoundError:
        pytest.skip("training.data is not importable in this test environment")

    csv_path = tmp_path / "coco2014_train_1kimg.tsv"
    csv_path.write_text("title\tfilepath\nhello world\timages/sample.jpg\n", encoding="utf-8")

    assert is_retrieval_dataset_path(str(csv_path)) is False


def test_imagenet_cache_uses_same_rendered_caption_as_training_dataset(tmp_path):
    try:
        from training.data import ImageNetCsvDataset
    except ModuleNotFoundError:
        pytest.skip("training.data is not importable in this test environment")

    module = load_cache_teacher_module()

    csv_path = tmp_path / "mini_imagenet_train.tsv"
    csv_path.write_text(
        "title\tfilepath\tclass\nsmall striped cat\timages/sample.jpg\ttabby cat\n",
        encoding="utf-8",
    )

    tokenizer_calls = []

    class DummyTokenizer:
        def __call__(self, texts):
            tokenizer_calls.append(list(texts))
            return [[1, 2, 0] for _ in texts]

    dataset = ImageNetCsvDataset(
        str(tmp_path),
        str(csv_path),
        transforms=lambda image: image,
        img_key="filepath",
        caption_key="title",
        sep="\t",
        tokenizer=DummyTokenizer(),
        deterministic_caption_templates=True,
    )

    cache_dataset = module.build_csv_like_dataset(
        input_filename=str(csv_path),
        data_root=str(tmp_path),
        preprocess_fn=lambda image: image,
        tokenizer=DummyTokenizer(),
        args=type(
            "Args",
            (),
            {
                "csv_separator": "\t",
                "csv_img_key": "filepath",
                "csv_caption_key": "title",
                "csv_caption_candidates_keys": "",
                "csv_metadata_keys": "",
            },
        )(),
        source_prefix="csv0:mini_imagenet_train.tsv",
    )

    rendered_caption = dataset.render_caption(0)
    assert cache_dataset.primary_captions[0] == rendered_caption
