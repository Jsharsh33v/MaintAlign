"""
MaintAlign — Input Validators
==============================

Centralized validation for user inputs (machine specs, problem instances,
CSV rows, solver parameters). Raises descriptive exceptions so callers
— the CLI, Streamlit UI, and tests — can surface clear error messages
instead of cryptic tracebacks from deep inside the solver.

Usage:
    from core.validators import validate_instance, InvalidInstanceError

    try:
        validate_instance(inst)
    except InvalidInstanceError as e:
        print(f"Bad instance: {e}")
"""

from __future__ import annotations

from typing import Any


# ── Custom exceptions ────────────────────────────────────────

class MaintAlignError(Exception):
    """Base class for all MaintAlign validation errors."""


class InvalidMachineSpecError(MaintAlignError):
    """Raised when a machine spec has invalid parameters."""


class InvalidInstanceError(MaintAlignError):
    """Raised when a problem instance is inconsistent or infeasible."""


class InvalidCSVRowError(MaintAlignError):
    """Raised when a CSV row cannot be parsed into a valid machine."""


class InvalidSolverParamsError(MaintAlignError):
    """Raised when solver parameters (time limit, repair factor, etc.) are bad."""


# ── Primitive validators ─────────────────────────────────────

def _require_positive_int(value: Any, field_name: str) -> int:
    """Cast to int and require > 0."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise InvalidMachineSpecError(
            f"{field_name}: expected integer, got {value!r}"
        )
    if v <= 0:
        raise InvalidMachineSpecError(
            f"{field_name}: must be > 0, got {v}"
        )
    return v


def _require_nonneg_int(value: Any, field_name: str) -> int:
    """Cast to int and require >= 0."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise InvalidMachineSpecError(
            f"{field_name}: expected integer, got {value!r}"
        )
    if v < 0:
        raise InvalidMachineSpecError(
            f"{field_name}: must be >= 0, got {v}"
        )
    return v


def _require_positive_float(value: Any, field_name: str) -> float:
    """Cast to float and require > 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InvalidMachineSpecError(
            f"{field_name}: expected number, got {value!r}"
        )
    if v <= 0:
        raise InvalidMachineSpecError(
            f"{field_name}: must be > 0, got {v}"
        )
    return v


# ── Machine-level validation ─────────────────────────────────

def validate_machine_spec(machine) -> None:
    """
    Validate one MachineSpec. Raises InvalidMachineSpecError on any problem.

    Checks:
      - name is non-empty string
      - duration, pm_cost, cm_cost, production_value are positive ints
      - weibull_beta > 0
      - weibull_eta > 0
      - max_interval >= duration (otherwise PM can never fit)
      - min_gap >= 0
      - cm_cost > pm_cost (otherwise PM is never economical)
      - repair_factor in (0, 1]
    """
    if not machine.name or not isinstance(machine.name, str):
        raise InvalidMachineSpecError(
            f"Machine id={getattr(machine, 'id', '?')}: name must be a non-empty string"
        )

    d = _require_positive_int(machine.maintenance_duration,
                              f"Machine '{machine.name}'.maintenance_duration")
    _require_positive_int(machine.pm_cost, f"Machine '{machine.name}'.pm_cost")
    cm = _require_positive_int(machine.cm_cost,
                               f"Machine '{machine.name}'.cm_cost")
    pm = machine.pm_cost
    _require_nonneg_int(machine.production_value,
                        f"Machine '{machine.name}'.production_value")
    _require_positive_float(machine.weibull_beta,
                            f"Machine '{machine.name}'.weibull_beta")
    _require_positive_float(machine.weibull_eta,
                            f"Machine '{machine.name}'.weibull_eta")

    w = _require_positive_int(machine.max_interval,
                              f"Machine '{machine.name}'.max_interval")
    if w < d:
        raise InvalidMachineSpecError(
            f"Machine '{machine.name}': max_interval ({w}) must be >= "
            f"maintenance_duration ({d}) or no PM can ever be scheduled"
        )

    _require_nonneg_int(machine.min_gap, f"Machine '{machine.name}'.min_gap")

    if cm <= pm:
        raise InvalidMachineSpecError(
            f"Machine '{machine.name}': cm_cost ({cm}) must exceed "
            f"pm_cost ({pm}); otherwise PM is never worthwhile"
        )

    rf = getattr(machine, 'repair_factor', 1.0)
    if not (0.0 < float(rf) <= 1.0):
        raise InvalidMachineSpecError(
            f"Machine '{machine.name}': repair_factor ({rf}) must be in (0, 1]"
        )


# ── Instance-level validation ────────────────────────────────

def validate_instance(instance) -> None:
    """
    Validate a ProblemInstance end-to-end. Raises InvalidInstanceError.

    Checks:
      - horizon > 0 and >= longest maintenance duration
      - num_technicians > 0
      - at least one machine
      - every machine passes validate_machine_spec
      - chain machine_ids all exist
      - no machine appears in two chains
      - blocked_periods are within [0, horizon)
    """
    if instance.horizon <= 0:
        raise InvalidInstanceError(
            f"horizon must be > 0, got {instance.horizon}"
        )

    if instance.num_technicians <= 0:
        raise InvalidInstanceError(
            f"num_technicians must be > 0, got {instance.num_technicians}"
        )

    if instance.num_machines == 0 or len(instance.machines) == 0:
        raise InvalidInstanceError("instance must contain at least one machine")

    if len(instance.machines) != instance.num_machines:
        raise InvalidInstanceError(
            f"num_machines ({instance.num_machines}) does not match "
            f"len(machines) ({len(instance.machines)})"
        )

    # Per-machine validation
    for m in instance.machines:
        validate_machine_spec(m)
        if m.maintenance_duration > instance.horizon:
            raise InvalidInstanceError(
                f"Machine '{m.name}' duration ({m.maintenance_duration}) "
                f"exceeds horizon ({instance.horizon})"
            )

    # Chain validation
    valid_ids = {m.id for m in instance.machines}
    seen_in_chain = set()
    for ch in instance.chains:
        for mid in ch.machine_ids:
            if mid not in valid_ids:
                raise InvalidInstanceError(
                    f"Chain '{ch.name}' references unknown machine_id={mid}"
                )
            if mid in seen_in_chain:
                raise InvalidInstanceError(
                    f"Machine_id={mid} appears in multiple chains "
                    f"(second one: '{ch.name}')"
                )
            seen_in_chain.add(mid)
        if ch.chain_value < 0:
            raise InvalidInstanceError(
                f"Chain '{ch.name}': chain_value must be >= 0, got {ch.chain_value}"
            )
        if ch.retooling_cost < 0:
            raise InvalidInstanceError(
                f"Chain '{ch.name}': retooling_cost must be >= 0, "
                f"got {ch.retooling_cost}"
            )

    # Blocked periods validation
    for t in getattr(instance, "blocked_periods", []):
        if not (0 <= t < instance.horizon):
            raise InvalidInstanceError(
                f"blocked_period {t} is outside [0, {instance.horizon})"
            )


# ── CSV row validation ───────────────────────────────────────

def validate_csv_row(row: list, row_num: int) -> None:
    """
    Validate a parsed CSV row before conversion to MachineSpec.
    Raises InvalidCSVRowError with row context.

    Expected 9 columns:
        name, duration, pm_cost, cm_cost, prod_value, beta, eta, max_interval, min_gap
    """
    if len(row) < 9:
        raise InvalidCSVRowError(
            f"Row {row_num}: expected 9 columns, got {len(row)}. Row: {row}"
        )

    if not row[0].strip():
        raise InvalidCSVRowError(f"Row {row_num}: machine name is empty")

    # Numeric fields — give clear context on which column failed
    numeric_fields = [
        (1, "duration", int),
        (2, "pm_cost", int),
        (3, "cm_cost", int),
        (4, "prod_value", int),
        (5, "beta", float),
        (6, "eta", float),
        (7, "max_interval", int),
        (8, "min_gap", int),
    ]
    for idx, name, caster in numeric_fields:
        val = row[idx].strip()
        try:
            caster(val)
        except ValueError:
            raise InvalidCSVRowError(
                f"Row {row_num} (machine='{row[0].strip()}'): column "
                f"'{name}' has invalid {caster.__name__} value {val!r}"
            )


# ── Solver parameter validation ──────────────────────────────

def validate_solver_params(time_limit_seconds: float = 60,
                            repair_factor: float = 1.0,
                            n_sims: int = 500) -> None:
    """Validate parameters passed to solve() or Monte Carlo."""
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise InvalidSolverParamsError(
            f"time_limit_seconds must be > 0, got {time_limit_seconds}"
        )
    if not (0.0 < repair_factor <= 1.0):
        raise InvalidSolverParamsError(
            f"repair_factor must be in (0, 1], got {repair_factor}"
        )
    if n_sims < 1:
        raise InvalidSolverParamsError(
            f"n_sims must be >= 1, got {n_sims}"
        )
