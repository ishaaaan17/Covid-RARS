from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


FULL_RELIABILITY_PROFILE = "full_reliability"
CAPACITY_INTERNAL_FUSION_PROFILE = "capacity_internal_cough_speech"


@dataclass(frozen=True)
class HSTWorkloadProfile:
    name: str
    primary_modalities: tuple[str, ...]
    secondary_modalities: tuple[str, ...]
    training_jobs_by_modality: Mapping[str, int]
    training_jobs_by_stage: Mapping[str, int]
    terminal_stage: str

    @property
    def total_training_jobs(self) -> int:
        return int(sum(self.training_jobs_by_modality.values()))


_PROFILES = {
    FULL_RELIABILITY_PROFILE: HSTWorkloadProfile(
        name=FULL_RELIABILITY_PROFILE,
        primary_modalities=("cough", "speech"),
        secondary_modalities=("breath",),
        training_jobs_by_modality=MappingProxyType(
            {"breath": 10, "cough": 25, "speech": 15}
        ),
        training_jobs_by_stage=MappingProxyType(
            {"internal_cv": 40, "split_policy_contrast": 8, "reverse_temporal": 2}
        ),
        terminal_stage="evidence_pack",
    ),
    CAPACITY_INTERNAL_FUSION_PROFILE: HSTWorkloadProfile(
        name=CAPACITY_INTERNAL_FUSION_PROFILE,
        primary_modalities=("cough", "speech"),
        secondary_modalities=(),
        training_jobs_by_modality=MappingProxyType({"cough": 10, "speech": 10}),
        training_jobs_by_stage=MappingProxyType({"internal_cv": 20}),
        terminal_stage="evidence_pack",
    ),
}
WORKLOAD_PROFILES: Mapping[str, HSTWorkloadProfile] = MappingProxyType(_PROFILES)


def get_hst_workload_profile(name: object) -> HSTWorkloadProfile:
    normalized = str(name or FULL_RELIABILITY_PROFILE).strip()
    try:
        return WORKLOAD_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown frozen HST workload profile: {normalized!r}") from exc


def workload_profile_from_scientific_config(
    scientific_config: Mapping[str, object],
) -> HSTWorkloadProfile:
    experiment = scientific_config.get("experiment")
    if not isinstance(experiment, Mapping):
        return get_hst_workload_profile(FULL_RELIABILITY_PROFILE)
    profile = get_hst_workload_profile(experiment.get("workload_profile"))
    if "primary_modalities" not in experiment and "secondary_modalities" not in experiment:
        return profile
    primary = tuple(str(value) for value in experiment.get("primary_modalities", ()))
    secondary = tuple(str(value) for value in experiment.get("secondary_modalities", ()))
    if primary != profile.primary_modalities or secondary != profile.secondary_modalities:
        raise ValueError(
            "HST experiment modalities disagree with the frozen workload profile"
        )
    return profile
