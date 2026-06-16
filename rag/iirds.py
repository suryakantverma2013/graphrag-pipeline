"""iiRDS controlled vocabulary (D9 / FR-4.2 / FR-4.2a).

`lifecycle_phase` and `information_type` are CLOSED enumerations validated
against these config-defined value sets (the diagram's value sets), rejecting or
normalizing out-of-vocabulary values. The lists are intentionally extensible to
the full iiRDS vocabulary later. `product` / `component` are OPEN free-text and
are MERGE'd in the graph (handled in the Neo4j write stage), not constrained here.
"""

from __future__ import annotations

from enum import Enum


class LifecyclePhase(str, Enum):
    INSTALLATION = "Installation"
    OPERATION = "Operation"
    SERVICE = "Service"
    REPAIR = "Repair"
    DISPOSAL = "Disposal"


class InformationType(str, Enum):
    PROCEDURE = "Procedure"
    CONCEPT = "Concept"
    WARNING = "Warning"
    SPECIFICATION = "Specification"
    MAINTENANCE_INTERVAL = "MaintenanceInterval"
    TROUBLESHOOTING = "Troubleshooting"


def _normalize(value: str, enum: type[Enum]) -> str | None:
    """Case-insensitive match of `value` to an enum member; None if out-of-vocab."""
    if value is None:
        return None
    target = value.strip().casefold()
    for member in enum:
        if member.value.casefold() == target:
            return member.value
    return None


def normalize_lifecycle_phase(value: str) -> str | None:
    """Return the canonical phase name, or None if out-of-vocabulary (FR-4.2a)."""
    return _normalize(value, LifecyclePhase)


def normalize_information_type(value: str) -> str | None:
    """Return the canonical info-type name, or None if out-of-vocabulary (FR-4.2a)."""
    return _normalize(value, InformationType)
