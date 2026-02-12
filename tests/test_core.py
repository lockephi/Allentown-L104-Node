"""
Tests for const.py — Universal Constants and God Code equation.
"""
import math
import pytest
from const import (
    UniversalConstants, GOD_CODE, PHI, HARMONIC_BASE,
    L104, OCTAVE_REF, INVARIANT, verify_conservation,
    god_code_at, grover_boost,
)


class TestUniversalConstants:
    """Test the UniversalConstants class methods and values."""

    def test_god_code_at_x0(self):
        """G(0) = 286^(1/φ_growth) × 2^(416/104) = 527.518..."""
        g0 = UniversalConstants.god_code(0)
        assert abs(g0 - GOD_CODE) < 1e-8

    def test_conservation_law(self):
        """G(X) × 2^(X/104) = INVARIANT for many X values."""
        for x in [-200, -100, -50, 0, 50, 100, 200, 416]:
            result = UniversalConstants.conservation_check(x)
            assert abs(result - INVARIANT) < 1e-8, f"Conservation failed at X={x}: {result}"

    def test_verify_conservation_function(self):
        assert verify_conservation(0) is True
        assert verify_conservation(100) is True
        assert verify_conservation(-50) is True

    def test_phi_values(self):
        """PHI and PHI_GROWTH should be reciprocals shifted by 1."""
        assert abs(UniversalConstants.PHI_GROWTH - (1 + UniversalConstants.PHI)) < 1e-12
        assert abs(UniversalConstants.PHI * UniversalConstants.PHI_GROWTH - 1) < 1e-12

    def test_factor_13(self):
        """286, 104, 416 all share factor 13."""
        assert HARMONIC_BASE % 13 == 0
        assert L104 % 13 == 0
        assert OCTAVE_REF % 13 == 0

    def test_god_code_monotone(self):
        """G(X) should decrease as X increases."""
        g0 = god_code_at(0)
        g100 = god_code_at(100)
        g416 = god_code_at(416)
        assert g0 > g100 > g416

    def test_web_api_url(self):
        url = UniversalConstants.web_api_url("health", port=8081)
        assert url == "http://localhost:8081/health"

    def test_quantum_amplify(self):
        """Quantum amplification should scale value up."""
        result = UniversalConstants.quantum_amplify(1.0, depth=3)
        assert result > 1.0

    def test_grover_boost(self):
        """Grover boost should amplify target index."""
        values = [1.0, 1.0, 1.0, 1.0]
        boosted = grover_boost(values, target_idx=0)
        assert len(boosted) == 4
        # Target should be amplified relative to others
        assert boosted[0] >= boosted[1]


class TestLogicCore:
    """Test the upgraded LogicCore."""

    def test_init(self):
        from logic_core import LogicCore
        core = LogicCore()
        assert core.god_code == GOD_CODE
        assert core.phi == PHI
        assert core.lattice_ratio == OCTAVE_REF / HARMONIC_BASE

    def test_adapt_to_core(self):
        from logic_core import LogicCore
        core = LogicCore()
        summary = core.adapt_to_core()
        assert "RESONANCE" in summary
        assert "LATTICE" in summary
        assert "527.518" in summary

    def test_ingest_data_state(self, tmp_path):
        """Test indexing on a temp directory."""
        from logic_core import LogicCore
        # Create some temp files
        (tmp_path / "test.py").write_text("print('hello')")
        (tmp_path / "test.txt").write_text("data")
        (tmp_path / "skip.pyc").write_bytes(b"\x00")

        core = LogicCore()
        core.ingest_data_state(str(tmp_path))
        assert len(core.manifold_memory) == 2  # .py and .txt, not .pyc

    def test_get_stats(self):
        from logic_core import LogicCore
        core = LogicCore()
        stats = core.get_stats()
        assert "data_points" in stats
        assert "god_code" in stats


class TestSQLitePool:
    """Test the SQLite connection pool."""

    def test_pool_basic(self, tmp_path):
        from l104_sqlite_pool import SQLitePool
        db_path = str(tmp_path / "test.db")
        pool = SQLitePool(db_path, max_connections=4)

        with pool.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)")
            conn.execute("INSERT INTO t (val) VALUES (?)", ("hello",))

        with pool.connection() as conn:
            row = conn.execute("SELECT val FROM t WHERE id=1").fetchone()
            assert row["val"] == "hello"

        pool.close()

    def test_pool_reuse(self, tmp_path):
        from l104_sqlite_pool import SQLitePool
        db_path = str(tmp_path / "test2.db")
        pool = SQLitePool(db_path, max_connections=2)

        # Use and return connection
        with pool.connection() as conn:
            conn.execute("CREATE TABLE t (id INTEGER)")

        # Second use should reuse the connection from pool
        with pool.connection() as conn:
            conn.execute("INSERT INTO t (id) VALUES (1)")

        assert pool._created <= 2
        pool.close()

    @pytest.mark.asyncio
    async def test_async_execute(self, tmp_path):
        from l104_sqlite_pool import SQLitePool
        db_path = str(tmp_path / "test3.db")
        pool = SQLitePool(db_path)

        with pool.connection() as conn:
            conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO t VALUES (1, 'async_test')")

        row = await pool.async_execute("SELECT name FROM t WHERE id=1", fetch="one")
        assert row["name"] == "async_test"
        pool.close()


class TestPublicNode:
    """Test the upgraded L104_public_node module."""

    def test_calculate_node_signature(self):
        from L104_public_node import calculate_node_signature
        sig = calculate_node_signature()
        assert isinstance(sig, str)
        assert len(sig) == 16

    def test_get_node_status(self):
        from L104_public_node import get_node_status
        status = get_node_status()
        assert "status" in status
        assert "invariant" in status
        assert status["invariant"] == GOD_CODE
        # No duplicate keys
        keys = list(status.keys())
        assert len(keys) == len(set(keys))

    def test_calculate_god_code(self):
        from L104_public_node import calculate_god_code
        g0 = calculate_god_code(0)
        assert abs(g0 - GOD_CODE) < 1e-8

    def test_verify_conservation(self):
        from L104_public_node import verify_conservation
        result = verify_conservation(0)
        assert result["conserved"] is True

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        from L104_public_node import heartbeat
        result = await heartbeat()
        assert "cycle" in result
        assert "resonance" in result
        assert result["status"] == "OK"


class TestSageLogicGate:
    """Test Sage Logic Gate + Quantum functions — cross-pollinated from Swift."""

    def test_sage_logic_gate_deterministic(self):
        from const import sage_logic_gate, GOD_CODE
        result = sage_logic_gate(GOD_CODE)
        assert isinstance(result, float)
        assert result > 0
        # Same input → same output (deterministic)
        assert sage_logic_gate(GOD_CODE) == result

    def test_sage_logic_gate_preserves_sign(self):
        from const import sage_logic_gate
        assert sage_logic_gate(100.0) > 0
        assert sage_logic_gate(-100.0) < 0
        assert sage_logic_gate(0.0) == 0.0

    def test_quantum_logic_gate(self):
        from const import quantum_logic_gate, GOD_CODE
        result = quantum_logic_gate(GOD_CODE, depth=3)
        assert isinstance(result, float)
        assert result > GOD_CODE  # Grover amplification should amplify

    def test_quantum_logic_gate_depth_scaling(self):
        from const import quantum_logic_gate
        r1 = quantum_logic_gate(100.0, depth=1)
        r3 = quantum_logic_gate(100.0, depth=3)
        r5 = quantum_logic_gate(100.0, depth=5)
        # Higher depth → more amplification
        assert r3 > r1
        assert r5 > r3

    def test_entangle_symmetric(self):
        from const import entangle
        ea, eb = entangle(0.5, 0.8)
        # Both should be between the two inputs (weighted average)
        assert min(0.5, 0.8) <= ea <= max(0.5, 0.8)
        assert min(0.5, 0.8) <= eb <= max(0.5, 0.8)

    def test_entangle_identical(self):
        from const import entangle
        ea, eb = entangle(1.0, 1.0)
        # Entangling identical values should return the same
        assert abs(ea - 1.0) < 1e-10
        assert abs(eb - 1.0) < 1e-10

    def test_chakra_align(self):
        from const import chakra_align, CHAKRA_FREQUENCIES
        aligned, idx = chakra_align(GOD_CODE)
        assert isinstance(aligned, float)
        assert 0 <= idx < len(CHAKRA_FREQUENCIES)

    def test_chakra_frequencies_count(self):
        from const import CHAKRA_FREQUENCIES
        assert len(CHAKRA_FREQUENCIES) == 7
        # Ascending order: root < crown
        assert CHAKRA_FREQUENCIES[0] < CHAKRA_FREQUENCIES[-1]

    def test_grover_boost_amplifies_target(self):
        from const import grover_boost
        values = [1.0, 1.0, 1.0, 1.0]
        boosted = grover_boost(values, target_idx=2)
        assert len(boosted) == 4
        # Target should be amplified relative to others
        assert abs(boosted[2]) >= abs(boosted[0])
