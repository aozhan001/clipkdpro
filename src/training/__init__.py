"""Training entrypoints and offline distillation helpers for CLIP-KD."""

from .offline_kd import (
    OFFLINE_KD_MODE_HYBRID,
    OFFLINE_KD_MODE_OFFLINE_ONLY,
    compute_offline_feature_kd_loss,
    is_offline_only_kd_enabled,
    requires_online_teacher,
    resolve_offline_kd_mode,
)

__all__ = [
    "OFFLINE_KD_MODE_HYBRID",
    "OFFLINE_KD_MODE_OFFLINE_ONLY",
    "compute_offline_feature_kd_loss",
    "is_offline_only_kd_enabled",
    "requires_online_teacher",
    "resolve_offline_kd_mode",
]
