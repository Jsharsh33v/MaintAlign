"""Tests for core.validators."""

import pytest

from core.instance import MachineSpec, ProductionChain, ProblemInstance
from core.validators import (
    validate_machine_spec,
    validate_instance,
    validate_csv_row,
    validate_solver_params,
    InvalidMachineSpecError,
    InvalidInstanceError,
    InvalidCSVRowError,
    InvalidSolverParamsError,
)


def _good_machine(mid=0, name="Lathe") -> MachineSpec:
    """Factory for a valid MachineSpec used across tests."""
    return MachineSpec(
        id=mid,
        name=name,
        maintenance_duration=2,
        pm_cost=400,
        cm_cost=3200,
        production_value=300,
        weibull_beta=2.5,
        weibull_eta=14,
        max_interval=14,
        min_gap=4,
    )


def _good_instance() -> ProblemInstance:
    """Factory for a valid ProblemInstance."""
    m0 = _good_machine(0, "Lathe")
    m1 = _good_machine(1, "Drill")
    return ProblemInstance(
        name="test",
        num_machines=2,
        num_technicians=1,
        horizon=20,
        machines=[m0, m1],
        chains=[],
    )


# ── Machine spec tests ───────────────────────────────────────

class TestValidateMachineSpec:

    def test_valid_spec_passes(self):
        validate_machine_spec(_good_machine())  # should not raise

    def test_empty_name_fails(self):
        m = _good_machine(name="")
        with pytest.raises(InvalidMachineSpecError, match="name"):
            validate_machine_spec(m)

    def test_zero_duration_fails(self):
        m = _good_machine()
        m.maintenance_duration = 0
        with pytest.raises(InvalidMachineSpecError, match="duration"):
            validate_machine_spec(m)

    def test_max_interval_less_than_duration_fails(self):
        m = _good_machine()
        m.max_interval = 1
        m.maintenance_duration = 2
        with pytest.raises(InvalidMachineSpecError, match="max_interval"):
            validate_machine_spec(m)

    def test_cm_cost_not_exceeding_pm_fails(self):
        m = _good_machine()
        m.pm_cost = 1000
        m.cm_cost = 500  # cheaper than PM — breaks the economics
        with pytest.raises(InvalidMachineSpecError, match="cm_cost"):
            validate_machine_spec(m)

    def test_repair_factor_out_of_range_fails(self):
        m = _good_machine()
        m.repair_factor = 1.5
        with pytest.raises(InvalidMachineSpecError, match="repair_factor"):
            validate_machine_spec(m)


# ── Instance tests ───────────────────────────────────────────

class TestValidateInstance:

    def test_valid_instance_passes(self):
        validate_instance(_good_instance())

    def test_zero_horizon_fails(self):
        inst = _good_instance()
        inst.horizon = 0
        with pytest.raises(InvalidInstanceError, match="horizon"):
            validate_instance(inst)

    def test_zero_technicians_fails(self):
        inst = _good_instance()
        inst.num_technicians = 0
        with pytest.raises(InvalidInstanceError, match="technicians"):
            validate_instance(inst)

    def test_empty_machines_fails(self):
        inst = _good_instance()
        inst.machines = []
        inst.num_machines = 0
        with pytest.raises(InvalidInstanceError, match="at least one machine"):
            validate_instance(inst)

    def test_chain_with_unknown_machine_fails(self):
        inst = _good_instance()
        inst.chains = [ProductionChain(
            id=0, name="bogus", machine_ids=[99],
            chain_value=100, retooling_cost=50,
        )]
        with pytest.raises(InvalidInstanceError, match="unknown machine_id"):
            validate_instance(inst)

    def test_duplicate_chain_membership_fails(self):
        inst = _good_instance()
        inst.chains = [
            ProductionChain(id=0, name="a", machine_ids=[0, 1],
                            chain_value=100, retooling_cost=50),
            ProductionChain(id=1, name="b", machine_ids=[1],  # 1 already used
                            chain_value=100, retooling_cost=50),
        ]
        with pytest.raises(InvalidInstanceError, match="multiple chains"):
            validate_instance(inst)


# ── CSV row tests ────────────────────────────────────────────

class TestValidateCSVRow:

    def test_valid_row_passes(self):
        row = ["Lathe", "2", "400", "3200", "300", "2.5", "14", "14", "4"]
        validate_csv_row(row, row_num=5)

    def test_too_few_columns_fails(self):
        row = ["Lathe", "2", "400"]
        with pytest.raises(InvalidCSVRowError, match="9 columns"):
            validate_csv_row(row, row_num=5)

    def test_empty_name_fails(self):
        row = ["", "2", "400", "3200", "300", "2.5", "14", "14", "4"]
        with pytest.raises(InvalidCSVRowError, match="name is empty"):
            validate_csv_row(row, row_num=5)

    def test_non_numeric_duration_fails(self):
        row = ["Lathe", "abc", "400", "3200", "300", "2.5", "14", "14", "4"]
        with pytest.raises(InvalidCSVRowError, match="duration"):
            validate_csv_row(row, row_num=5)


# ── Solver param tests ───────────────────────────────────────

class TestValidateSolverParams:

    def test_good_params_pass(self):
        validate_solver_params(time_limit_seconds=30, repair_factor=0.8, n_sims=500)

    def test_zero_time_limit_fails(self):
        with pytest.raises(InvalidSolverParamsError, match="time_limit"):
            validate_solver_params(time_limit_seconds=0)

    def test_repair_factor_zero_fails(self):
        with pytest.raises(InvalidSolverParamsError, match="repair_factor"):
            validate_solver_params(repair_factor=0.0)

    def test_repair_factor_above_one_fails(self):
        with pytest.raises(InvalidSolverParamsError, match="repair_factor"):
            validate_solver_params(repair_factor=1.1)

    def test_zero_sims_fails(self):
        with pytest.raises(InvalidSolverParamsError, match="n_sims"):
            validate_solver_params(n_sims=0)
