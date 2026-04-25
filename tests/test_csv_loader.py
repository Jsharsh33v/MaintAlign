"""Tests for utils.csv_loader."""

import os
import tempfile

import pytest

from utils.csv_loader import load_instance, load_machines_csv

GOOD_MACHINES_CSV = """#CONFIG, test_shop, 2, 30
name,duration,pm_cost,cm_cost,prod_value,beta,eta,max_interval,min_gap
Lathe,2,400,3200,300,2.5,14,14,4
Drill,1,150,900,120,2.0,10,10,3
"""

GOOD_CHAINS_CSV = """chain_name,machine_names,chain_value,retooling_cost
MainLine,Lathe;Drill,500,200
"""

BAD_MACHINES_CSV_MISSING_COLS = """#CONFIG, bad_shop, 2, 30
name,duration,pm_cost
Lathe,2,400
"""


@pytest.fixture
def tmp_csv_pair():
    """Write good machines + chains CSVs to temp files."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as mf:
        mf.write(GOOD_MACHINES_CSV)
        m_path = mf.name
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as cf:
        cf.write(GOOD_CHAINS_CSV)
        c_path = cf.name
    yield m_path, c_path
    os.unlink(m_path)
    os.unlink(c_path)


class TestLoadMachinesCSV:

    def test_loads_valid_csv(self, tmp_csv_pair):
        m_path, _ = tmp_csv_pair
        name, K, H, machines = load_machines_csv(m_path)
        assert name == "test_shop"
        assert K == 2
        assert H == 30
        assert len(machines) == 2
        assert machines[0].name == "Lathe"
        assert machines[1].name == "Drill"

    def test_raises_on_empty_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("")
            path = f.name
        try:
            with pytest.raises(ValueError, match="No valid machines"):
                load_machines_csv(path)
        finally:
            os.unlink(path)

    def test_skips_malformed_rows(self):
        content = GOOD_MACHINES_CSV + "BrokenRow,only,three,cols\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            _, _, _, machines = load_machines_csv(path)
            # The broken row should be skipped, not crash
            assert len(machines) == 2
        finally:
            os.unlink(path)


class TestLoadInstance:

    def test_loads_with_chains(self, tmp_csv_pair):
        m_path, c_path = tmp_csv_pair
        inst = load_instance(m_path, c_path)
        assert inst.num_machines == 2
        assert inst.horizon == 30
        assert len(inst.chains) == 1
        assert inst.chains[0].name == "MainLine"

    def test_loads_without_chains(self, tmp_csv_pair):
        m_path, _ = tmp_csv_pair
        inst = load_instance(m_path)
        assert inst.num_machines == 2
        assert len(inst.chains) == 0
