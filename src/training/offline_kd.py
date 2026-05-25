from __future__ import annotations

try:
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in environments without torch
    F = None


OFFLINE_KD_MODE_HYBRID = "hybrid"
OFFLINE_KD_MODE_OFFLINE_ONLY = "offline_only"
OFFLINE_KD_MODES = (
    OFFLINE_KD_MODE_HYBRID,
    OFFLINE_KD_MODE_OFFLINE_ONLY,
)
OFFLINE_TEXT_CANDIDATE_STRATEGY_DISABLED = "disabled"
OFFLINE_TEXT_CANDIDATE_STRATEGY_MEAN = "mean"
OFFLINE_TEXT_CANDIDATE_STRATEGY_MAX = "max"
OFFLINE_TEXT_CANDIDATE_STRATEGY_SOFTMAX = "softmax"
OFFLINE_TEXT_CANDIDATE_STRATEGY_TOP1 = "top1"
OFFLINE_TEXT_CANDIDATE_STRATEGIES = (
    OFFLINE_TEXT_CANDIDATE_STRATEGY_DISABLED,
    OFFLINE_TEXT_CANDIDATE_STRATEGY_MEAN,
    OFFLINE_TEXT_CANDIDATE_STRATEGY_MAX,
    OFFLINE_TEXT_CANDIDATE_STRATEGY_SOFTMAX,
    OFFLINE_TEXT_CANDIDATE_STRATEGY_TOP1,
)


def resolve_offline_kd_mode(args):
    mode = getattr(args, "offline_kd_mode", OFFLINE_KD_MODE_HYBRID)
    if mode is None:
        return OFFLINE_KD_MODE_HYBRID
    return str(mode)


def is_offline_only_kd_enabled(args):
    return (
        bool(getattr(args, "use_offline_teacher_cache", False)) and
        resolve_offline_kd_mode(args) == OFFLINE_KD_MODE_OFFLINE_ONLY
    )


def requires_online_teacher(args):
    return not is_offline_only_kd_enabled(args)


def compute_offline_feature_kd_loss(student_features, teacher_features, loss_type="mse"):
    if F is None:
        raise ImportError("compute_offline_feature_kd_loss requires torch to be installed.")

    student_features = F.normalize(student_features, dim=-1)
    teacher_features = F.normalize(teacher_features, dim=-1)
    if loss_type == "mse":
        return F.mse_loss(student_features, teacher_features)
    if loss_type == "cosine":
        return (1 - F.cosine_similarity(student_features, teacher_features, dim=-1)).mean()
    raise ValueError(f"Unsupported offline KD loss type: {loss_type}")


def resolve_offline_text_candidate_strategy(args):
    strategy = getattr(
        args,
        "offline_text_candidate_strategy",
        OFFLINE_TEXT_CANDIDATE_STRATEGY_DISABLED,
    )
    if strategy is None:
        return OFFLINE_TEXT_CANDIDATE_STRATEGY_DISABLED
    return str(strategy)


def is_candidate_text_kd_enabled(args):
    return (
        bool(getattr(args, "use_offline_teacher_cache", False)) and
        float(getattr(args, "lambda_text_kd_candidates", 0.0)) > 0.0 and
        resolve_offline_text_candidate_strategy(args) != OFFLINE_TEXT_CANDIDATE_STRATEGY_DISABLED
    )
