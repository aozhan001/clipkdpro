from __future__ import annotations

import logging
import math
import random

try:
    import torch
    from torch.utils.data import Sampler
    from torch.utils.data.distributed import DistributedSampler
except ImportError:  # pragma: no cover - exercised only in environments without torch
    torch = None

    class Sampler(object):
        pass

    class DistributedSampler(object):
        pass


def validate_keep_ratio(keep_ratio):
    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(f"sample_keep_ratio must be in (0, 1], but got {keep_ratio}.")
    return keep_ratio


def compute_num_kept_samples(num_samples, keep_ratio):
    keep_ratio = validate_keep_ratio(keep_ratio)
    if num_samples <= 0:
        return 0
    if keep_ratio >= 1.0:
        return num_samples
    return max(1, int(math.floor(num_samples * keep_ratio)))


def get_metadata_value(metadata_entry, *keys):
    for key in keys:
        if key in metadata_entry and metadata_entry[key] is not None:
            return metadata_entry[key]
    return None


def get_similarity_values(metadata):
    similarities = []
    for index, entry in enumerate(metadata):
        value = get_metadata_value(entry, "teacher_image_text_similarity", "image_text_similarity")
        if value is None:
            raise ValueError(
                "Sample filtering strategy requires image-text similarity metadata, "
                f"but cache metadata at index {index} does not provide it."
            )
        similarities.append(float(value))
    return similarities


def get_entropy_values(metadata):
    entropies = []
    for entry in metadata:
        value = get_metadata_value(entry, "teacher_entropy", "entropy")
        if value is None:
            return None
        entropies.append(float(value))
    return entropies


METRIC_KEY_ALIASES = {
    "similarity": ("teacher_image_text_similarity", "image_text_similarity"),
    "entropy": ("teacher_entropy", "entropy"),
    "difficulty": ("difficulty_score",),
    "density": ("density_score", "sample_density", "image_density"),
    "caption_quality": ("caption_quality", "caption_score", "caption_quality_score"),
    "disagreement": ("teacher_disagreement", "caption_disagreement", "disagreement"),
}


DEFAULT_SAMPLE_METRIC_WEIGHTS = {
    "similarity": 1.0,
    "entropy": 0.5,
    "disagreement": 0.5,
    "density": 0.25,
    "caption_quality": 0.5,
    "difficulty": 0.0,
}


def clamp01(value):
    return min(1.0, max(0.0, float(value)))


def normalize_values(values):
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [0.5 for _ in values]
    scale = maximum - minimum
    return [(float(value) - minimum) / scale for value in values]


def get_metric_values(metadata, metric_name, required=False):
    aliases = METRIC_KEY_ALIASES[metric_name]
    values = []
    missing_indices = []
    for index, entry in enumerate(metadata):
        value = get_metadata_value(entry, *aliases)
        if value is None:
            missing_indices.append(index)
            values.append(0.0)
        else:
            values.append(float(value))
    if missing_indices and required:
        raise ValueError(
            f"Sample filtering strategy requires metric '{metric_name}', but metadata is missing it "
            f"for cache entries such as index {missing_indices[0]}."
        )
    return values, bool(missing_indices)


def resolve_metric_weights(args=None):
    weights = dict(DEFAULT_SAMPLE_METRIC_WEIGHTS)
    if args is None:
        return weights
    for metric_name in DEFAULT_SAMPLE_METRIC_WEIGHTS:
        arg_name = f"sample_weight_{metric_name}"
        if hasattr(args, arg_name):
            weights[metric_name] = float(getattr(args, arg_name))
    return weights


def log_missing_metric_warning(metric_name):
    logging.warning(
        "Sample filtering metric '%s' is missing from some cache metadata entries. "
        "Those missing values will be treated as 0.0. You can precompute this metric into the cache metadata "
        "for more faithful filtering.",
        metric_name,
    )


def rank_indices_by_scores(scores, keep_count, seed=0, epoch=0):
    if keep_count >= len(scores):
        return list(range(len(scores)))
    rng = random.Random(seed + epoch + 17)
    tie_breakers = [rng.random() for _ in scores]
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: (scores[idx], tie_breakers[idx]),
        reverse=True,
    )
    return ranked_indices[:keep_count]


def select_top_metric_indices(metadata, metric_name, keep_ratio, seed=0, epoch=0):
    values, missing = get_metric_values(metadata, metric_name, required=False)
    if missing:
        log_missing_metric_warning(metric_name)
    keep_count = compute_num_kept_samples(len(values), keep_ratio)
    scores = normalize_values(values)
    return rank_indices_by_scores(scores, keep_count, seed=seed, epoch=epoch)


def compute_weighted_scores(metadata, args=None):
    weights = resolve_metric_weights(args)
    scores = [0.0 for _ in metadata]
    used_metric_count = 0
    for metric_name, weight in weights.items():
        if math.isclose(weight, 0.0):
            continue
        values, missing = get_metric_values(metadata, metric_name, required=False)
        if missing:
            log_missing_metric_warning(metric_name)
        normalized = normalize_values(values)
        for index, value in enumerate(normalized):
            scores[index] += weight * value
        used_metric_count += 1
    if used_metric_count == 0:
        similarities = normalize_values(get_similarity_values(metadata))
        return similarities
    return scores


def compute_curriculum_weighted_scores(metadata, args=None, epoch=0, curriculum_epoch=0):
    progress = clamp01(float(epoch) / max(1.0, float(curriculum_epoch)))
    weights = resolve_metric_weights(args)
    metric_values = {}
    for metric_name, weight in weights.items():
        if math.isclose(weight, 0.0):
            continue
        values, missing = get_metric_values(metadata, metric_name, required=False)
        if missing:
            log_missing_metric_warning(metric_name)
        metric_values[metric_name] = normalize_values(values)

    if not metric_values:
        return compute_weighted_scores(metadata, args=args)

    easy_coefficients = {
        "similarity": 1.0,
        "caption_quality": 1.0,
        "density": 1.0,
        "entropy": -1.0,
        "disagreement": -1.0,
        "difficulty": -1.0,
    }
    hard_coefficients = {
        "similarity": 0.35,
        "caption_quality": 0.35,
        "density": 0.15,
        "entropy": 1.0,
        "disagreement": 1.0,
        "difficulty": 1.0,
    }

    scores = [0.0 for _ in metadata]
    for metric_name, normalized_values in metric_values.items():
        weight = weights[metric_name]
        easy_weight = weight * easy_coefficients.get(metric_name, 0.0)
        hard_weight = weight * hard_coefficients.get(metric_name, 0.0)
        for index, value in enumerate(normalized_values):
            easy_score = easy_weight * value
            hard_score = hard_weight * value
            scores[index] += (1.0 - progress) * easy_score + progress * hard_score
    return scores


def dedupe_preserve_order(indices):
    output = []
    seen = set()
    for index in indices:
        if index in seen:
            continue
        output.append(index)
        seen.add(index)
    return output


def select_random_indices(num_samples, keep_ratio, seed=0, epoch=0):
    keep_count = compute_num_kept_samples(num_samples, keep_ratio)
    if keep_count >= num_samples:
        return list(range(num_samples))
    rng = random.Random(seed + epoch)
    return rng.sample(list(range(num_samples)), keep_count)


def select_top_similarity_indices(metadata, keep_ratio):
    similarities = get_similarity_values(metadata)
    keep_count = compute_num_kept_samples(len(similarities), keep_ratio)
    ranked_indices = sorted(range(len(similarities)), key=lambda idx: similarities[idx], reverse=True)
    return ranked_indices[:keep_count]


def select_middle_similarity_indices(metadata, keep_ratio):
    similarities = get_similarity_values(metadata)
    keep_count = compute_num_kept_samples(len(similarities), keep_ratio)
    ranked_indices = sorted(range(len(similarities)), key=lambda idx: similarities[idx])
    start = max(0, (len(ranked_indices) - keep_count) // 2)
    return ranked_indices[start:start + keep_count]


def select_high_entropy_indices(metadata, keep_ratio, seed=0, epoch=0):
    entropies = get_entropy_values(metadata)
    if entropies is None:
        logging.warning(
            "Sample filtering strategy 'high_entropy' requires teacher_entropy metadata. "
            "Falling back to random filtering."
        )
        return select_random_indices(len(metadata), keep_ratio, seed=seed, epoch=epoch)

    keep_count = compute_num_kept_samples(len(entropies), keep_ratio)
    ranked_indices = sorted(range(len(entropies)), key=lambda idx: entropies[idx], reverse=True)
    return ranked_indices[:keep_count]


def select_curriculum_indices(metadata, keep_ratio, epoch=0, curriculum_epoch=0):
    similarities = get_similarity_values(metadata)
    keep_count = compute_num_kept_samples(len(similarities), keep_ratio)
    ranked_indices = sorted(range(len(similarities)), key=lambda idx: similarities[idx], reverse=True)

    progress = min(1.0, max(0.0, float(epoch)) / max(1.0, float(curriculum_epoch)))
    hard_count = int(round(progress * keep_count))
    easy_count = keep_count - hard_count

    selected = []
    if easy_count > 0:
        selected.extend(ranked_indices[:easy_count])
    if hard_count > 0:
        selected.extend(ranked_indices[-hard_count:])
    selected = dedupe_preserve_order(selected)

    if len(selected) < keep_count:
        for index in ranked_indices:
            if index not in selected:
                selected.append(index)
            if len(selected) == keep_count:
                break
    return selected[:keep_count]


def select_weighted_indices(metadata, keep_ratio, seed=0, epoch=0, args=None):
    scores = compute_weighted_scores(metadata, args=args)
    keep_count = compute_num_kept_samples(len(scores), keep_ratio)
    return rank_indices_by_scores(scores, keep_count, seed=seed, epoch=epoch)


def select_curriculum_weighted_indices(metadata, keep_ratio, seed=0, epoch=0, curriculum_epoch=0, args=None):
    scores = compute_curriculum_weighted_scores(
        metadata,
        args=args,
        epoch=epoch,
        curriculum_epoch=curriculum_epoch,
    )
    keep_count = compute_num_kept_samples(len(scores), keep_ratio)
    return rank_indices_by_scores(scores, keep_count, seed=seed, epoch=epoch)


def select_sample_indices(metadata, strategy, keep_ratio, seed=0, epoch=0, curriculum_epoch=0, args=None):
    strategy = strategy or "random"
    keep_ratio = validate_keep_ratio(keep_ratio)
    if len(metadata) == 0:
        return []

    if strategy == "random":
        return select_random_indices(len(metadata), keep_ratio, seed=seed, epoch=epoch)
    if strategy == "top_similarity":
        return select_top_similarity_indices(metadata, keep_ratio)
    if strategy == "middle_similarity":
        return select_middle_similarity_indices(metadata, keep_ratio)
    if strategy == "high_entropy":
        return select_high_entropy_indices(metadata, keep_ratio, seed=seed, epoch=epoch)
    if strategy == "high_density":
        return select_top_metric_indices(metadata, "density", keep_ratio, seed=seed, epoch=epoch)
    if strategy == "high_caption_quality":
        return select_top_metric_indices(metadata, "caption_quality", keep_ratio, seed=seed, epoch=epoch)
    if strategy == "high_disagreement":
        return select_top_metric_indices(metadata, "disagreement", keep_ratio, seed=seed, epoch=epoch)
    if strategy == "curriculum":
        return select_curriculum_indices(
            metadata,
            keep_ratio,
            epoch=epoch,
            curriculum_epoch=curriculum_epoch,
        )
    if strategy == "weighted":
        return select_weighted_indices(metadata, keep_ratio, seed=seed, epoch=epoch, args=args)
    if strategy == "curriculum_weighted":
        return select_curriculum_weighted_indices(
            metadata,
            keep_ratio,
            seed=seed,
            epoch=epoch,
            curriculum_epoch=curriculum_epoch,
            args=args,
        )
    raise ValueError(f"Unsupported sample filter strategy: {strategy}")


def is_sample_filter_requested(args, is_train=False):
    if not is_train:
        return False
    strategy = getattr(args, "sample_filter_strategy", "random")
    keep_ratio = float(getattr(args, "sample_keep_ratio", 1.0))
    return keep_ratio < 1.0 or strategy != "random"


def should_enable_sample_filtering(args, is_train=False):
    if not is_sample_filter_requested(args, is_train=is_train):
        return False
    if not getattr(args, "use_offline_teacher_cache", False):
        logging.warning(
            "Sample filtering was requested, but offline teacher cache is disabled. "
            "Filtering is skipped because it requires cache metadata."
        )
        return False
    strategy = getattr(args, "sample_filter_strategy", "random")
    keep_ratio = float(getattr(args, "sample_keep_ratio", 1.0))
    if strategy == "random" and keep_ratio >= 1.0:
        logging.info(
            "Sample filtering strategy '%s' is configured with keep ratio 1.0, so filtering is skipped.",
            strategy,
        )
        return False
    return True


def should_shuffle_filtered_samples(strategy):
    return (strategy or "random") == "random"


class OfflineFilteredSampler(Sampler):
    def __init__(
        self,
        dataset,
        strategy="random",
        keep_ratio=1.0,
        seed=0,
        curriculum_epoch=0,
        shuffle=True,
        args=None,
    ):
        if torch is None:
            raise ImportError("OfflineFilteredSampler requires torch to be installed.")
        self.dataset = dataset
        self.metadata = dataset.offline_teacher_cache.metadata
        self.strategy = strategy
        self.keep_ratio = validate_keep_ratio(keep_ratio)
        self.seed = seed
        self.curriculum_epoch = curriculum_epoch
        self.shuffle = shuffle
        self.args = args
        self.epoch = 0
        self.selected_count = compute_num_kept_samples(len(self.metadata), self.keep_ratio)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_filtered_indices(self):
        return select_sample_indices(
            self.metadata,
            strategy=self.strategy,
            keep_ratio=self.keep_ratio,
            seed=self.seed,
            epoch=self.epoch,
            curriculum_epoch=self.curriculum_epoch,
            args=self.args,
        )

    def __iter__(self):
        indices = list(self.get_filtered_indices())
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch + 1009)
            rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.selected_count


class OfflineFilteredDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        strategy="random",
        keep_ratio=1.0,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
        curriculum_epoch=0,
        args=None,
    ):
        if torch is None:
            raise ImportError("OfflineFilteredDistributedSampler requires torch to be installed.")
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.metadata = dataset.offline_teacher_cache.metadata
        self.strategy = strategy
        self.keep_ratio = validate_keep_ratio(keep_ratio)
        self.curriculum_epoch = curriculum_epoch
        self.args = args
        self.selected_count = compute_num_kept_samples(len(self.metadata), self.keep_ratio)
        self.num_samples = self._compute_num_samples(self.selected_count)
        self.total_size = self.num_samples * self.num_replicas

    def _compute_num_samples(self, filtered_count):
        if filtered_count == 0:
            return 0
        if self.drop_last and filtered_count % self.num_replicas != 0:
            return math.ceil((filtered_count - self.num_replicas) / self.num_replicas)
        return math.ceil(filtered_count / self.num_replicas)

    def get_filtered_indices(self):
        return select_sample_indices(
            self.metadata,
            strategy=self.strategy,
            keep_ratio=self.keep_ratio,
            seed=self.seed,
            epoch=self.epoch,
            curriculum_epoch=self.curriculum_epoch,
            args=self.args,
        )

    def __iter__(self):
        indices = list(self.get_filtered_indices())
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[idx] for idx in permutation]

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0 and len(indices) > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def build_offline_sample_filter_sampler(dataset, args, is_train=False):
    if not should_enable_sample_filtering(args, is_train=is_train):
        return None
    if not hasattr(dataset, "offline_teacher_cache"):
        logging.warning(
            "Sample filtering requires a dataset with offline teacher cache metadata. "
            "Filtering is skipped for dataset type %s.",
            type(dataset).__name__,
        )
        return None

    strategy = getattr(args, "sample_filter_strategy", "random")
    keep_ratio = float(getattr(args, "sample_keep_ratio", 1.0))
    curriculum_epoch = int(getattr(args, "curriculum_epoch", 0))
    seed = int(getattr(args, "seed", 0))
    shuffle = should_shuffle_filtered_samples(strategy)

    if getattr(args, "distributed", False):
        return OfflineFilteredDistributedSampler(
            dataset,
            strategy=strategy,
            keep_ratio=keep_ratio,
            num_replicas=getattr(args, "world_size", None),
            rank=getattr(args, "rank", None),
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
            curriculum_epoch=curriculum_epoch,
            args=args,
        )
    return OfflineFilteredSampler(
        dataset,
        strategy=strategy,
        keep_ratio=keep_ratio,
        seed=seed,
        curriculum_epoch=curriculum_epoch,
        shuffle=shuffle,
        args=args,
    )
