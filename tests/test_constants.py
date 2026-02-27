"""
Unit tests for experiment target constants.
"""

from constants import (
    EXPERIMENT_TARGETS, RUN_TO_EXPERIMENT, TARGET_ACCURACY, VALID_RUNS
)


class TestExperimentTargets:
    """Tests for per-experiment accuracy targets."""

    def test_six_experiments_defined(self):
        """Test that exactly 6 experiments are defined."""
        assert len(EXPERIMENT_TARGETS) == 6

    def test_experiment_indices(self):
        """Test that experiments are indexed 0-5."""
        assert set(EXPERIMENT_TARGETS.keys()) == {0, 1, 2, 3, 4, 5}

    def test_all_targets_above_fifty_percent(self):
        """Test that all targets are above 50%."""
        for target in EXPERIMENT_TARGETS.values():
            assert target > 0.50

    def test_mean_target_matches_global(self):
        """Test that mean of per-experiment targets matches global target."""
        mean = sum(EXPERIMENT_TARGETS.values()) / len(EXPERIMENT_TARGETS)
        assert abs(mean - TARGET_ACCURACY) < 0.05

    def test_expected_target_values(self):
        """Test the specific target values from the spec."""
        assert EXPERIMENT_TARGETS[0] == 0.59
        assert EXPERIMENT_TARGETS[1] == 0.57
        assert EXPERIMENT_TARGETS[2] == 0.71
        assert EXPERIMENT_TARGETS[3] == 0.60
        assert EXPERIMENT_TARGETS[4] == 0.59
        assert EXPERIMENT_TARGETS[5] == 0.67


class TestRunToExperiment:
    """Tests for run-to-experiment mapping."""

    def test_all_valid_runs_mapped(self):
        """Test that all valid runs have an experiment mapping."""
        for run in VALID_RUNS:
            assert run in RUN_TO_EXPERIMENT

    def test_experiment_indices_valid(self):
        """Test that all mapped experiment indices exist in targets."""
        for exp_idx in RUN_TO_EXPERIMENT.values():
            assert exp_idx in EXPERIMENT_TARGETS

    def test_consecutive_run_pairs(self):
        """Test that consecutive run pairs map to same experiment."""
        assert RUN_TO_EXPERIMENT[3] == RUN_TO_EXPERIMENT[4]
        assert RUN_TO_EXPERIMENT[5] == RUN_TO_EXPERIMENT[6]
        assert RUN_TO_EXPERIMENT[7] == RUN_TO_EXPERIMENT[8]
        assert RUN_TO_EXPERIMENT[9] == RUN_TO_EXPERIMENT[10]
        assert RUN_TO_EXPERIMENT[11] == RUN_TO_EXPERIMENT[12]
        assert RUN_TO_EXPERIMENT[13] == RUN_TO_EXPERIMENT[14]
