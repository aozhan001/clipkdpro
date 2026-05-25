import logging
import os

import torch
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_collate


def build_csv_dataset_source_prefix(input_filename, dataset_idx=0):
    return f"csv{dataset_idx}:{os.path.basename(input_filename)}"


def build_csv_dataset_sample_ids(dataset, source_prefix):
    if hasattr(dataset, "class_name"):
        return [
            f"{source_prefix}::{idx}::{dataset.class_name[idx]}::{dataset.images[idx]}"
            for idx in range(len(dataset))
        ]
    return [
        f"{source_prefix}::{idx}::{dataset.images[idx]}"
        for idx in range(len(dataset))
    ]


def build_sequential_sample_ids(prefix, size):
    return [f"{prefix}::{idx}" for idx in range(size)]


def default_difficulty_score(image_text_similarity):
    if image_text_similarity is None:
        return None
    return float(1.0 - image_text_similarity)


def normalize_cache_metadata_entry(entry, index):
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise ValueError(f"Cache metadata entry at index {index} must be a dict, but got {type(entry)}.")

    metadata = dict(entry)
    metadata.setdefault("sample_id", None)
    metadata.setdefault("caption_length", None)
    metadata.setdefault("image_text_similarity", None)
    metadata.setdefault("caption_quality", None)
    metadata.setdefault("density_score", None)
    metadata.setdefault("teacher_entropy", None)
    metadata.setdefault("caption_disagreement", metadata.get("teacher_disagreement"))
    if "difficulty_score" not in metadata or metadata["difficulty_score"] is None:
        metadata["difficulty_score"] = default_difficulty_score(metadata.get("image_text_similarity"))
    return metadata


class OfflineTeacherCache:
    def __init__(self, cache_path, map_location="cpu"):
        self.cache_path = cache_path
        payload = torch.load(cache_path, map_location=map_location)
        if not isinstance(payload, dict):
            raise ValueError(f"Teacher cache at {cache_path} must be a dict saved by torch.save.")

        if "image_features" not in payload:
            raise ValueError(f"Teacher cache at {cache_path} is missing required field 'image_features'.")
        if "text_features" not in payload:
            raise ValueError(f"Teacher cache at {cache_path} is missing required field 'text_features'.")

        self.image_features = payload["image_features"]
        self.text_features = payload["text_features"]
        self.candidate_text_features = payload.get("candidate_text_features")
        self.candidate_text_feature_mask = payload.get("candidate_text_feature_mask")
        if not isinstance(self.image_features, torch.Tensor):
            raise ValueError(f"'image_features' in {cache_path} must be a torch.Tensor.")
        if not isinstance(self.text_features, torch.Tensor):
            raise ValueError(f"'text_features' in {cache_path} must be a torch.Tensor.")
        if self.image_features.ndim != 2 or self.text_features.ndim != 2:
            raise ValueError(
                f"Teacher cache tensors must be 2D [N, D], got {self.image_features.shape} and {self.text_features.shape}."
            )
        if self.image_features.shape[0] != self.text_features.shape[0]:
            raise ValueError(
                f"Teacher cache feature count mismatch: {self.image_features.shape[0]} image features vs "
                f"{self.text_features.shape[0]} text features."
            )
        if self.candidate_text_features is not None:
            if not isinstance(self.candidate_text_features, torch.Tensor):
                raise ValueError(f"'candidate_text_features' in {cache_path} must be a torch.Tensor.")
            if self.candidate_text_features.ndim != 3:
                raise ValueError(
                    f"'candidate_text_features' in {cache_path} must be 3D [N, K, D], got {self.candidate_text_features.shape}."
                )
            if self.candidate_text_features.shape[0] != self.text_features.shape[0]:
                raise ValueError(
                    f"Candidate text feature count mismatch: {self.candidate_text_features.shape[0]} candidate rows vs "
                    f"{self.text_features.shape[0]} cached samples."
                )
            if self.candidate_text_features.shape[2] != self.text_features.shape[1]:
                raise ValueError(
                    f"Candidate text feature dim mismatch: {self.candidate_text_features.shape[2]} vs "
                    f"{self.text_features.shape[1]}."
                )
        if self.candidate_text_feature_mask is not None:
            if not isinstance(self.candidate_text_feature_mask, torch.Tensor):
                raise ValueError(f"'candidate_text_feature_mask' in {cache_path} must be a torch.Tensor.")
            if self.candidate_text_feature_mask.ndim != 2:
                raise ValueError(
                    f"'candidate_text_feature_mask' in {cache_path} must be 2D [N, K], got {self.candidate_text_feature_mask.shape}."
                )
            if self.candidate_text_features is None:
                raise ValueError(
                    f"'candidate_text_feature_mask' is present in {cache_path}, but 'candidate_text_features' is missing."
                )
            expected_shape = self.candidate_text_features.shape[:2]
            if tuple(self.candidate_text_feature_mask.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Candidate text feature mask shape mismatch: {tuple(self.candidate_text_feature_mask.shape)} vs "
                    f"{tuple(expected_shape)}."
                )
            self.candidate_text_feature_mask = self.candidate_text_feature_mask.to(dtype=torch.bool)
        elif self.candidate_text_features is not None:
            self.candidate_text_feature_mask = torch.ones(
                self.candidate_text_features.shape[:2],
                dtype=torch.bool,
            )

        raw_metadata = payload.get("metadata")
        if raw_metadata is None:
            logging.warning("Teacher cache %s has no metadata field. Sample-id lookup will be unavailable.", cache_path)
            raw_metadata = [{} for _ in range(self.image_features.shape[0])]
        if len(raw_metadata) != self.image_features.shape[0]:
            raise ValueError(
                f"Teacher cache metadata length mismatch: {len(raw_metadata)} metadata entries vs "
                f"{self.image_features.shape[0]} cached samples."
            )

        self.metadata = [
            normalize_cache_metadata_entry(entry, index)
            for index, entry in enumerate(raw_metadata)
        ]
        self.config = payload.get("config", {})
        self.summary = payload.get("summary", {})
        self.num_samples = self.image_features.shape[0]

        sample_ids = [entry.get("sample_id") for entry in self.metadata]
        self.has_complete_sample_ids = all(sample_id is not None for sample_id in sample_ids)
        self.sample_id_to_index = {}
        if self.has_complete_sample_ids:
            for index, sample_id in enumerate(sample_ids):
                if sample_id in self.sample_id_to_index:
                    raise ValueError(
                        f"Teacher cache {cache_path} contains duplicate sample_id '{sample_id}'."
                    )
                self.sample_id_to_index[sample_id] = index
        elif any(sample_id is not None for sample_id in sample_ids):
            logging.warning(
                "Teacher cache %s has partial sample_id coverage. Falling back to index-based lookup only.",
                cache_path,
            )

    def __len__(self):
        return self.num_samples

    def get_by_index(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Teacher cache index {index} is out of range for cache of size {self.num_samples}.")
        metadata = self.metadata[index]
        return {
            "teacher_image_features": self.image_features[index],
            "teacher_text_features": self.text_features[index],
            "candidate_teacher_text_features": None if self.candidate_text_features is None else self.candidate_text_features[index],
            "candidate_teacher_text_feature_mask": None if self.candidate_text_feature_mask is None else self.candidate_text_feature_mask[index],
            "image_text_similarity": metadata.get("image_text_similarity"),
            "caption_length": metadata.get("caption_length"),
            "difficulty_score": metadata.get("difficulty_score"),
            "caption_quality": metadata.get("caption_quality"),
            "density_score": metadata.get("density_score"),
            "teacher_entropy": metadata.get("teacher_entropy"),
            "caption_disagreement": metadata.get("caption_disagreement"),
            "metadata": metadata,
            "sample_id": metadata.get("sample_id"),
            "cache_index": index,
        }

    def get_by_sample_id(self, sample_id):
        if not self.has_complete_sample_ids:
            raise ValueError(
                f"Teacher cache {self.cache_path} does not have complete sample_id metadata, so sample_id lookup is unavailable."
            )
        if sample_id not in self.sample_id_to_index:
            raise KeyError(f"sample_id '{sample_id}' was not found in teacher cache {self.cache_path}.")
        return self.get_by_index(self.sample_id_to_index[sample_id])


class SampleIDDatasetWrapper(Dataset):
    def __init__(self, dataset, sample_ids):
        self.dataset = dataset
        self.sample_ids = list(sample_ids)
        if len(self.sample_ids) != len(self.dataset):
            raise ValueError(
                f"Sample-id wrapper length mismatch: dataset has {len(self.dataset)} samples but "
                f"{len(self.sample_ids)} sample_ids were provided."
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_sample_id(self, index):
        return self.sample_ids[index]

    def __getattr__(self, item):
        return getattr(self.dataset, item)


class OfflineTeacherSample:
    def __init__(self, sample, offline_teacher_cache):
        self.sample = sample
        self.offline_teacher_cache = offline_teacher_cache

    def __len__(self):
        return len(self.sample)

    def __iter__(self):
        return iter(self.sample)

    def __getitem__(self, index):
        return self.sample[index]


class OfflineTeacherBatch:
    def __init__(self, batch, offline_teacher_cache):
        self.batch = tuple(batch)
        self.offline_teacher_cache = offline_teacher_cache

    def __len__(self):
        return len(self.batch)

    def __iter__(self):
        return iter(self.batch)

    def __getitem__(self, index):
        return self.batch[index]


def collect_dataset_sample_ids(dataset):
    if isinstance(dataset, SampleIDDatasetWrapper):
        return list(dataset.sample_ids)
    if isinstance(dataset, OfflineTeacherCacheDatasetWrapper):
        return collect_dataset_sample_ids(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        sample_ids = []
        for child_dataset in dataset.datasets:
            child_sample_ids = collect_dataset_sample_ids(child_dataset)
            if child_sample_ids is None:
                return None
            sample_ids.extend(child_sample_ids)
        return sample_ids
    return None


class OfflineTeacherCacheDatasetWrapper(Dataset):
    def __init__(self, dataset, cache):
        self.dataset = dataset
        self.offline_teacher_cache = cache if isinstance(cache, OfflineTeacherCache) else OfflineTeacherCache(cache)
        self.sample_ids = collect_dataset_sample_ids(dataset)

        if self.sample_ids is not None and len(self.sample_ids) != len(self.dataset):
            raise ValueError(
                f"Dataset sample-id count mismatch: dataset has {len(self.dataset)} samples but "
                f"{len(self.sample_ids)} sample_ids were collected."
            )

        if len(self.offline_teacher_cache) != len(self.dataset):
            raise ValueError(
                f"Teacher cache size mismatch: dataset has {len(self.dataset)} samples but cache "
                f"{self.offline_teacher_cache.cache_path} has {len(self.offline_teacher_cache)} samples."
            )

        self.lookup_by_sample_id = False
        if self.sample_ids is not None and self.offline_teacher_cache.has_complete_sample_ids:
            missing_sample_ids = [
                sample_id for sample_id in self.sample_ids
                if sample_id not in self.offline_teacher_cache.sample_id_to_index
            ]
            if missing_sample_ids:
                preview = ", ".join(missing_sample_ids[:3])
                raise ValueError(
                    f"Teacher cache {self.offline_teacher_cache.cache_path} is missing dataset sample_ids. "
                    f"Examples: {preview}"
                )
            self.lookup_by_sample_id = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        cache_entry = self.get_teacher_cache(index=index)
        return OfflineTeacherSample(sample, cache_entry)

    def get_sample_id(self, index):
        if self.sample_ids is None:
            raise ValueError("This dataset wrapper does not have sample_id metadata.")
        return self.sample_ids[index]

    def get_teacher_cache(self, index=None, sample_id=None):
        if sample_id is not None:
            return self.offline_teacher_cache.get_by_sample_id(sample_id)
        if index is None:
            raise ValueError("Either index or sample_id must be provided to get_teacher_cache().")
        if self.lookup_by_sample_id:
            return self.offline_teacher_cache.get_by_sample_id(self.sample_ids[index])
        return self.offline_teacher_cache.get_by_index(index)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def collate_fn(self, batch):
        samples = [sample.sample for sample in batch]
        collated = default_collate(samples)
        if not isinstance(collated, (list, tuple)):
            raise ValueError("Offline teacher cache collate expects sequence-like dataset samples.")

        cache_entries = [sample.offline_teacher_cache for sample in batch]
        teacher_image_features = torch.stack(
            [entry["teacher_image_features"] for entry in cache_entries], dim=0
        )
        teacher_text_features = torch.stack(
            [entry["teacher_text_features"] for entry in cache_entries], dim=0
        )
        offline_teacher_cache = {
            "teacher_image_features": teacher_image_features,
            "teacher_text_features": teacher_text_features,
            "candidate_teacher_text_features": None,
            "candidate_teacher_text_feature_mask": None,
            "image_text_similarity": [entry.get("image_text_similarity") for entry in cache_entries],
            "caption_length": [entry.get("caption_length") for entry in cache_entries],
            "difficulty_score": [entry.get("difficulty_score") for entry in cache_entries],
            "caption_quality": [entry.get("caption_quality") for entry in cache_entries],
            "density_score": [entry.get("density_score") for entry in cache_entries],
            "teacher_entropy": [entry.get("teacher_entropy") for entry in cache_entries],
            "caption_disagreement": [entry.get("caption_disagreement") for entry in cache_entries],
            "sample_id": [entry.get("sample_id") for entry in cache_entries],
            "metadata": [entry.get("metadata") for entry in cache_entries],
            "cache_index": [entry.get("cache_index") for entry in cache_entries],
        }
        if cache_entries and cache_entries[0].get("candidate_teacher_text_features") is not None:
            offline_teacher_cache["candidate_teacher_text_features"] = torch.stack(
                [entry["candidate_teacher_text_features"] for entry in cache_entries], dim=0
            )
            offline_teacher_cache["candidate_teacher_text_feature_mask"] = torch.stack(
                [entry["candidate_teacher_text_feature_mask"] for entry in cache_entries], dim=0
            )
        return OfflineTeacherBatch(collated, offline_teacher_cache)
