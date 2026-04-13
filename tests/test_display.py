"""
Unit tests for display.py module.

Tests all terminal output functions for correct content,
format, and branching logic (target achieved vs. not, time limit, etc.).
"""

import numpy as np


class TestSection:
    """Tests for the section() function."""

    def test_prints_title(self, capsys):
        from display import section
        section("My Title")
        out = capsys.readouterr().out
        assert "My Title" in out

    def test_prints_separators(self, capsys):
        from display import section, SEP
        section("X")
        out = capsys.readouterr().out
        assert SEP in out


class TestPrintHeader:
    """Tests for print_header()."""

    def test_contains_subject_and_run(self, capsys):
        from display import print_header
        print_header(subject=4, run=14, mode="train", pipeline="csp_lda", run_type="imagery")
        out = capsys.readouterr().out
        assert "4" in out
        assert "14" in out
        assert "train" in out
        assert "csp_lda" in out
        assert "imagery" in out


class TestPrintDataInfo:
    """Tests for print_data_info()."""

    def test_prints_shape(self, capsys):
        from display import print_data_info
        X = np.zeros((30, 64, 160))
        y = np.array([1] * 15 + [2] * 15)
        print_data_info(X, y)
        out = capsys.readouterr().out
        assert "(30, 64, 160)" in out

    def test_prints_class_distribution(self, capsys):
        from display import print_data_info
        X = np.zeros((4, 64, 160))
        y = np.array([1, 1, 2, 2])
        print_data_info(X, y)
        out = capsys.readouterr().out
        assert "1" in out
        assert "2" in out


class TestPrintCvResult:
    """Tests for print_cv_result()."""

    def test_target_achieved(self, capsys):
        from display import print_cv_result
        scores = np.array([0.8, 0.85, 0.9])
        print_cv_result(scores, "csp_lda", target=0.60)
        out = capsys.readouterr().out
        assert "ACHIEVED" in out

    def test_target_not_achieved(self, capsys):
        from display import print_cv_result
        scores = np.array([0.4, 0.45, 0.5])
        print_cv_result(scores, "csp_lda", target=0.60)
        out = capsys.readouterr().out
        assert "NOT achieved" in out

    def test_prints_mean_and_std(self, capsys):
        from display import print_cv_result
        scores = np.array([0.7, 0.75, 0.8])
        print_cv_result(scores, "csp_lda", target=0.60)
        out = capsys.readouterr().out
        assert "Mean accuracy" in out

    def test_prints_pipeline_name(self, capsys):
        from display import print_cv_result
        scores = np.array([0.7, 0.75])
        print_cv_result(scores, "my_pipeline", target=0.60)
        out = capsys.readouterr().out
        assert "my_pipeline" in out


class TestPrintPipelineResult:
    """Tests for print_pipeline_result()."""

    def test_prints_name_and_mean(self, capsys):
        from display import print_pipeline_result
        scores = np.array([0.7, 0.8])
        print_pipeline_result("csp_svm", scores)
        out = capsys.readouterr().out
        assert "csp_svm" in out
        assert "Mean accuracy" in out


class TestPrintBestPipeline:
    """Tests for print_best_pipeline()."""

    def test_prints_name_and_accuracy(self, capsys):
        from display import print_best_pipeline
        print_best_pipeline("csp_lda", 0.8234)
        out = capsys.readouterr().out
        assert "csp_lda" in out
        assert "0.8234" in out


class TestPrintTrainingSummary:
    """Tests for print_training_summary()."""

    def test_target_achieved(self, capsys):
        from display import print_training_summary
        scores = np.array([0.8, 0.85])
        print_training_summary(scores, exp_idx=1, exp_target=0.60)
        out = capsys.readouterr().out
        assert "ACHIEVED" in out

    def test_target_not_reached(self, capsys):
        from display import print_training_summary
        scores = np.array([0.4, 0.45])
        print_training_summary(scores, exp_idx=2, exp_target=0.60)
        out = capsys.readouterr().out
        assert "not reached" in out

    def test_exp_idx_none(self, capsys):
        from display import print_training_summary
        scores = np.array([0.8, 0.85])
        print_training_summary(scores, exp_idx=None, exp_target=0.60)
        out = capsys.readouterr().out
        assert "TRAINING COMPLETE" in out


class TestPrintPredictionSummary:
    """Tests for print_prediction_summary()."""

    def _make_results(self, accuracy: float, within_limit: bool) -> dict:
        return {
            'accuracy': accuracy,
            'avg_time': 0.05,
            'max_time': 0.1,
            'within_time_limit': within_limit,
        }

    def test_target_achieved(self, capsys):
        from display import print_prediction_summary
        results = self._make_results(0.9, True)
        print_prediction_summary(results, exp_idx=1, exp_target=0.60)
        out = capsys.readouterr().out
        assert "ACHIEVED" in out

    def test_target_not_reached(self, capsys):
        from display import print_prediction_summary
        results = self._make_results(0.4, True)
        print_prediction_summary(results, exp_idx=1, exp_target=0.60)
        out = capsys.readouterr().out
        assert "not reached" in out

    def test_time_limit_passed(self, capsys):
        from display import print_prediction_summary
        results = self._make_results(0.8, True)
        print_prediction_summary(results, exp_idx=1, exp_target=0.60)
        out = capsys.readouterr().out
        assert "PASSED" in out

    def test_time_limit_failed(self, capsys):
        from display import print_prediction_summary
        results = self._make_results(0.8, False)
        print_prediction_summary(results, exp_idx=1, exp_target=0.60)
        out = capsys.readouterr().out
        assert "FAILED" in out


class TestPrintRealtimeHeader:
    """Tests for print_realtime_header()."""

    def test_prints_n_epochs(self, capsys):
        from display import print_realtime_header
        print_realtime_header(n_epochs=45)
        out = capsys.readouterr().out
        assert "45" in out

    def test_prints_max_time(self, capsys):
        from display import print_realtime_header
        print_realtime_header(n_epochs=10, max_time=3.0)
        out = capsys.readouterr().out
        assert "3.0" in out


class TestPrintRealtimeEpoch:
    """Tests for print_realtime_epoch()."""

    def test_correct_prediction(self, capsys):
        from display import print_realtime_epoch
        print_realtime_epoch(i=0, pred=1, true=1, elapsed=0.05)
        out = capsys.readouterr().out
        assert "True" in out
        assert "[SLOW]" not in out

    def test_wrong_prediction(self, capsys):
        from display import print_realtime_epoch
        print_realtime_epoch(i=1, pred=2, true=1, elapsed=0.05)
        out = capsys.readouterr().out
        assert "False" in out

    def test_slow_flag(self, capsys):
        from display import print_realtime_epoch
        print_realtime_epoch(i=2, pred=1, true=1, elapsed=5.0, max_time=2.0)
        out = capsys.readouterr().out
        assert "[SLOW]" in out

    def test_epoch_index_formatted(self, capsys):
        from display import print_realtime_epoch
        print_realtime_epoch(i=3, pred=1, true=2, elapsed=0.1)
        out = capsys.readouterr().out
        assert "03" in out


class TestPrintRealtimeSummary:
    """Tests for print_realtime_summary()."""

    def test_time_limit_passed(self, capsys):
        from display import print_realtime_summary
        print_realtime_summary(
            accuracy=0.8, n_epochs=10,
            avg_time=0.05, max_pred_time=0.1,
            within_limit=True
        )
        out = capsys.readouterr().out
        assert "PASSED" in out

    def test_time_limit_failed(self, capsys):
        from display import print_realtime_summary
        print_realtime_summary(
            accuracy=0.8, n_epochs=10,
            avg_time=0.05, max_pred_time=3.0,
            within_limit=False
        )
        out = capsys.readouterr().out
        assert "FAILED" in out

    def test_correct_count(self, capsys):
        from display import print_realtime_summary
        print_realtime_summary(
            accuracy=0.6, n_epochs=10,
            avg_time=0.05, max_pred_time=0.1,
            within_limit=True
        )
        out = capsys.readouterr().out
        # 60% of 10 = 6 correct
        assert "6/10" in out

    def test_prints_accuracy(self, capsys):
        from display import print_realtime_summary
        print_realtime_summary(
            accuracy=0.75, n_epochs=8,
            avg_time=0.03, max_pred_time=0.08,
            within_limit=True
        )
        out = capsys.readouterr().out
        assert "0.7500" in out


class TestPrintModelInfo:
    """Tests for print_model_info()."""

    def test_prints_model_path(self, capsys):
        from display import print_model_info
        print_model_info("models/model_s4_r14_csp_lda.pkl", {})
        out = capsys.readouterr().out
        assert "models/model_s4_r14_csp_lda.pkl" in out

    def test_prints_cv_mean_when_present(self, capsys):
        from display import print_model_info
        print_model_info("model.pkl", {"subject": 4, "runs": [14], "cv_mean": 0.8234})
        out = capsys.readouterr().out
        assert "0.8234" in out

    def test_handles_missing_metadata(self, capsys):
        from display import print_model_info
        print_model_info("model.pkl", {})
        out = capsys.readouterr().out
        assert "unknown" in out

    def test_no_cv_mean_when_absent(self, capsys):
        from display import print_model_info
        print_model_info("model.pkl", {"subject": 1})
        out = capsys.readouterr().out
        assert "CV accuracy" not in out
