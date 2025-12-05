"""
Unit tests for preprocess.py module.

Tests EEG data preprocessing functions including:
- Run type identification
- Data shape validation
"""

import pytest
import numpy as np


class TestGetRunType:
    """Tests for get_run_type function."""

    def test_baseline_runs(self):
        """Test baseline run identification."""
        from preprocess import get_run_type

        assert get_run_type(1) == 'baseline'
        assert get_run_type(2) == 'baseline'

    def test_left_right_runs(self):
        """Test left/right motor imagery run identification."""
        from preprocess import get_run_type

        # Motor execution runs
        for run in [3, 7, 11]:
            assert get_run_type(run) == 'left_right'

        # Motor imagery runs
        for run in [4, 8, 12]:
            assert get_run_type(run) == 'left_right'

    def test_hands_feet_runs(self):
        """Test hands/feet motor imagery run identification."""
        from preprocess import get_run_type

        # Motor execution runs
        for run in [5, 9, 13]:
            assert get_run_type(run) == 'hands_feet'

        # Motor imagery runs
        for run in [6, 10, 14]:
            assert get_run_type(run) == 'hands_feet'

    def test_invalid_run_raises_error(self):
        """Test that invalid run numbers raise ValueError."""
        from preprocess import get_run_type

        with pytest.raises(ValueError, match="Invalid run number"):
            get_run_type(0)

        with pytest.raises(ValueError, match="Invalid run number"):
            get_run_type(15)

        with pytest.raises(ValueError, match="Invalid run number"):
            get_run_type(-1)


class TestEventIds:
    """Tests for event ID dictionaries."""

    def test_event_id_hands_feet(self):
        """Test hands/feet event ID mapping."""
        from preprocess import EVENT_ID_HANDS_FEET

        assert 'T1' in EVENT_ID_HANDS_FEET
        assert 'T2' in EVENT_ID_HANDS_FEET
        assert EVENT_ID_HANDS_FEET['T1'] == 1
        assert EVENT_ID_HANDS_FEET['T2'] == 2

    def test_event_id_left_right(self):
        """Test left/right event ID mapping."""
        from preprocess import EVENT_ID_LEFT_RIGHT

        assert 'T1' in EVENT_ID_LEFT_RIGHT
        assert 'T2' in EVENT_ID_LEFT_RIGHT
        assert EVENT_ID_LEFT_RIGHT['T1'] == 1
        assert EVENT_ID_LEFT_RIGHT['T2'] == 2


class TestDataPath:
    """Tests for data path functions."""

    def test_get_data_path_returns_string(self):
        """Test that get_data_path returns a string path."""
        from preprocess import get_data_path

        path = get_data_path()
        assert isinstance(path, str)
        assert len(path) > 0


class TestPreprocessSubjectValidation:
    """Tests for preprocess_subject validation logic."""

    def test_mixed_run_types_raises_error(self, mocker):
        """Test that mixing run types raises an error."""
        from preprocess import preprocess_subject

        # Mock load_raw_data to avoid actual data download
        mock_raw = mocker.MagicMock()
        mocker.patch('preprocess.load_raw_data', return_value=mock_raw)
        mocker.patch('preprocess.filter_raw', return_value=mock_raw)

        # Runs 4 (left_right) and 6 (hands_feet) should raise error
        with pytest.raises(ValueError, match="All runs must be of the same type"):
            preprocess_subject(1, [4, 6])

    def test_baseline_runs_raises_error(self, mocker):
        """Test that baseline runs cannot be used for classification."""
        from preprocess import preprocess_subject

        mock_raw = mocker.MagicMock()
        mocker.patch('preprocess.load_raw_data', return_value=mock_raw)
        mocker.patch('preprocess.filter_raw', return_value=mock_raw)

        with pytest.raises(ValueError, match="Cannot preprocess baseline runs"):
            preprocess_subject(1, [1])


class TestFilterRawParameters:
    """Tests for filter_raw function parameters."""

    def test_default_filter_frequencies(self):
        """Test default filter frequencies are motor imagery range."""
        from preprocess import filter_raw
        import inspect

        sig = inspect.signature(filter_raw)
        assert sig.parameters['l_freq'].default == 7.0
        assert sig.parameters['h_freq'].default == 30.0


class TestExtractEpochsParameters:
    """Tests for extract_epochs function parameters."""

    def test_default_epoch_timing(self):
        """Test default epoch timing parameters."""
        from preprocess import extract_epochs
        import inspect

        sig = inspect.signature(extract_epochs)
        assert sig.parameters['tmin'].default == 0.0
        assert sig.parameters['tmax'].default == 3.0
        assert sig.parameters['baseline'].default is None
