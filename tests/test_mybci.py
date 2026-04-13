"""
Unit tests for mybci.py CLI entry point.

Tests argument parsing, validation, and mode dispatch without
downloading any EEG data from Physionet.
"""

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestParseArgs:
    """Tests for parse_args function."""

    def test_basic_train_args(self):
        """Test default train arguments are parsed correctly."""
        from mybci import parse_args

        with patch('sys.argv', ['mybci.py', '4', '14', 'train']):
            args = parse_args()

        assert args.subject == 4
        assert args.run == 14
        assert args.mode == 'train'
        assert args.pipeline == 'csp_lda'
        assert args.cv == 5
        assert args.compare is False

    def test_basic_predict_args(self):
        """Test default predict arguments are parsed correctly."""
        from mybci import parse_args

        with patch('sys.argv', ['mybci.py', '1', '6', 'predict']):
            args = parse_args()

        assert args.subject == 1
        assert args.run == 6
        assert args.mode == 'predict'

    def test_custom_pipeline(self):
        """Test --pipeline flag is captured."""
        from mybci import parse_args

        with patch('sys.argv', ['mybci.py', '4', '14', 'train', '--pipeline', 'psd_lda']):
            args = parse_args()

        assert args.pipeline == 'psd_lda'

    def test_compare_flag(self):
        """Test --compare flag is captured."""
        from mybci import parse_args

        with patch('sys.argv', ['mybci.py', '4', '14', 'train', '--compare']):
            args = parse_args()

        assert args.compare is True

    def test_quiet_flag(self):
        """Test --quiet flag is captured."""
        from mybci import parse_args

        with patch('sys.argv', ['mybci.py', '4', '14', 'train', '--quiet']):
            args = parse_args()

        assert args.quiet is True

    def test_no_plot_flag(self):
        """Test --no-plot flag is captured."""
        from mybci import parse_args

        with patch('sys.argv', ['mybci.py', '4', '14', 'train', '--no-plot']):
            args = parse_args()

        assert args.no_plot is True

    def test_cv_option(self):
        """Test --cv option is parsed as integer."""
        from mybci import parse_args

        with patch('sys.argv', ['mybci.py', '1', '6', 'train', '--cv', '3']):
            args = parse_args()

        assert args.cv == 3


class TestValidateArgs:
    """Tests for validate_args function."""

    def test_valid_args_no_exit(self):
        """Test that valid subject and run do not trigger sys.exit."""
        from mybci import validate_args

        args = argparse.Namespace(subject=1, run=6)
        validate_args(args)  # should not raise

    def test_subject_too_low_exits(self):
        """Test sys.exit when subject < 1."""
        from mybci import validate_args

        args = argparse.Namespace(subject=0, run=6)
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_subject_too_high_exits(self):
        """Test sys.exit when subject > 109."""
        from mybci import validate_args

        args = argparse.Namespace(subject=110, run=6)
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_invalid_run_exits(self):
        """Test sys.exit for a run number outside 3-14."""
        from mybci import validate_args

        args = argparse.Namespace(subject=1, run=15)
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_boundary_subjects_valid(self):
        """Test boundary subject values (1 and 109) are accepted."""
        from mybci import validate_args

        for subject in [1, 109]:
            args = argparse.Namespace(subject=subject, run=6)
            validate_args(args)

    def test_boundary_runs_valid(self):
        """Test boundary run values (3 and 14) are accepted."""
        from mybci import validate_args

        for run in [3, 14]:
            args = argparse.Namespace(subject=1, run=run)
            validate_args(args)


class TestModeTrain:
    """Tests for mode_train function."""

    def test_returns_zero_on_success(self, small_synthetic_data):
        """Test that mode_train returns 0 on a successful run."""
        from mybci import mode_train

        X, y = small_synthetic_data
        mock_scores = np.array([0.7, 0.75, 0.65])

        args = argparse.Namespace(
            subject=1, run=6, pipeline='csp_lda', cv=2,
            compare=False, model_dir='models', quiet=False,
            no_plot=True, save_plots=False, n_components=4, reg=None
        )

        with patch('mybci.train_subject', return_value=(MagicMock(), mock_scores)), \
             patch('mybci.display.print_training_summary'):
            result = mode_train(args)

        assert result == 0

    def test_compare_mode_selects_best_pipeline(self, small_synthetic_data):
        """Test that --compare selects the best pipeline for final training."""
        from mybci import mode_train

        X, y = small_synthetic_data
        mock_scores = np.array([0.7, 0.75])

        compare_results = {
            'csp_lda': {'mean': 0.80, 'std': 0.05, 'scores': mock_scores},
            'psd_lda': {'mean': 0.65, 'std': 0.05, 'scores': mock_scores},
        }

        args = argparse.Namespace(
            subject=1, run=6, pipeline='csp_lda', cv=2,
            compare=True, model_dir='models', quiet=False,
            no_plot=True, save_plots=False, n_components=4, reg=None
        )

        with patch('mybci.preprocess_subject', return_value=(X, y, MagicMock())), \
             patch('mybci.compare_pipelines', return_value=compare_results), \
             patch('mybci.display.print_data_info'), \
             patch('mybci.train_subject', return_value=(MagicMock(), mock_scores)), \
             patch('mybci.display.print_training_summary'):
            mode_train(args)

        assert args.pipeline == 'csp_lda'


class TestModePredict:
    """Tests for mode_predict function."""

    def test_returns_zero_on_success(self):
        """Test that mode_predict returns 0 on a successful run."""
        from mybci import mode_predict

        args = argparse.Namespace(
            subject=1, run=6, pipeline='csp_lda',
            model_path=None, model_dir='models', quiet=False
        )

        prediction_result = {
            'accuracy': 0.8,
            'avg_time': 0.1,
            'max_time': 0.2,
            'within_time_limit': True,
        }
        with patch('mybci.run_prediction', return_value=prediction_result), \
             patch('mybci.display.print_prediction_summary'):
            result = mode_predict(args)

        assert result == 0


class TestMain:
    """Tests for main function error handling."""

    def test_keyboard_interrupt_returns_one(self):
        """Test that KeyboardInterrupt is caught and returns 1."""
        from mybci import main

        with patch('sys.argv', ['mybci.py', '1', '6', 'train']), \
             patch('mybci.validate_args'), \
             patch('mybci.display.print_header'), \
             patch('mybci.mode_train', side_effect=KeyboardInterrupt):
            result = main()

        assert result == 1

    def test_file_not_found_returns_one(self):
        """Test that FileNotFoundError is caught and returns 1."""
        from mybci import main

        with patch('sys.argv', ['mybci.py', '1', '6', 'predict']), \
             patch('mybci.validate_args'), \
             patch('mybci.display.print_header'), \
             patch('mybci.mode_predict', side_effect=FileNotFoundError("model.pkl")):
            result = main()

        assert result == 1

    def test_unexpected_exception_returns_one(self):
        """Test that unexpected exceptions are caught and return 1."""
        from mybci import main

        with patch('sys.argv', ['mybci.py', '1', '6', 'train']), \
             patch('mybci.validate_args'), \
             patch('mybci.display.print_header'), \
             patch('mybci.mode_train', side_effect=RuntimeError("unexpected")):
            result = main()

        assert result == 1
