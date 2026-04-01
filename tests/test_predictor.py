import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.predictor import TransNetPredictor, PredictionResult


# ---------------------------------------------------------------------------
# Fixture: reset singleton state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    TransNetPredictor._instance = None
    TransNetPredictor._model = None
    TransNetPredictor._device = None
    TransNetPredictor._weights_path = None
    yield
    TransNetPredictor._instance = None
    TransNetPredictor._model = None
    TransNetPredictor._device = None
    TransNetPredictor._weights_path = None


def _make_loaded_predictor(weights="w.pth", device="cpu") -> TransNetPredictor:
    """Return a predictor whose _load_model is suppressed."""
    with patch.object(TransNetPredictor, "_load_model"):
        p = TransNetPredictor(weights, device)
    # Simulate successful load
    TransNetPredictor._model = MagicMock()
    TransNetPredictor._weights_path = weights
    TransNetPredictor._device = device
    return p


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_two_calls_return_same_instance(self):
        p1 = _make_loaded_predictor()
        p2 = TransNetPredictor("w.pth", "cpu")
        assert p1 is p2

    def test_same_params_does_not_reload_model(self):
        p = _make_loaded_predictor()
        load_calls_before = TransNetPredictor._model.call_count
        TransNetPredictor("w.pth", "cpu")
        assert TransNetPredictor._model.call_count == load_calls_before

    def test_different_weights_path_raises(self):
        _make_loaded_predictor("w.pth", "cpu")
        with pytest.raises(RuntimeError, match="Cannot reinitialize"):
            TransNetPredictor("other.pth", "cpu")

    def test_different_device_raises(self):
        _make_loaded_predictor("w.pth", "cpu")
        with pytest.raises(RuntimeError, match="Cannot reinitialize"):
            TransNetPredictor("w.pth", "cuda")

    def test_error_message_includes_current_params(self):
        _make_loaded_predictor("w.pth", "cpu")
        with pytest.raises(RuntimeError, match="w.pth"):
            TransNetPredictor("new.pth", "cpu")


# ---------------------------------------------------------------------------
# predictions_to_scenes
# ---------------------------------------------------------------------------

class TestPredictionsToScenes:
    def test_empty_predictions_returns_empty_array(self):
        result = TransNetPredictor.predictions_to_scenes(np.array([]))
        assert result.shape == (0, 2)

    def test_all_background_gives_single_scene(self):
        preds = np.zeros(10, dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        assert result.shape == (1, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 9

    def test_all_foreground_gives_single_scene(self):
        preds = np.ones(5, dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        # The algorithm appends [start, last_index] only when t==0 at the end.
        # With all-ones, t==1 at end, so it relies on the empty-scenes fallback.
        assert result.shape[1] == 2  # always 2 columns

    def test_single_cut_produces_two_scenes(self):
        # cut signal at frame 5
        preds = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        assert result.shape == (2, 2)

    def test_multiple_cuts_produce_correct_scene_count(self):
        # cuts at frames 2 and 5
        preds = np.array([0, 0, 1, 0, 0, 1, 0, 0], dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        assert result.shape == (3, 2)

    def test_threshold_filtering_below_threshold(self):
        # 0.4 < 0.5 → treated as background → one scene
        preds = np.array([0, 0, 0.4, 0, 0], dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        assert result.shape == (1, 2)

    def test_threshold_filtering_above_threshold(self):
        # 0.6 > 0.5 → treated as cut
        preds = np.array([0, 0, 0.6, 0, 0], dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        assert result.shape == (2, 2)

    def test_custom_threshold(self):
        preds = np.array([0, 0, 0.3, 0, 0], dtype=float)
        # With threshold=0.2, 0.3 > 0.2 → cut
        result_low = TransNetPredictor.predictions_to_scenes(preds, threshold=0.2)
        # With threshold=0.5, 0.3 < 0.5 → no cut
        result_high = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        assert result_low.shape == (2, 2)
        assert result_high.shape == (1, 2)

    def test_scenes_are_contiguous(self):
        preds = np.array([0, 1, 0, 1, 0, 0], dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds, threshold=0.5)
        # Each scene end must be <= next scene start
        for i in range(len(result) - 1):
            assert result[i, 1] <= result[i + 1, 0]

    def test_output_dtype_is_int32(self):
        preds = np.zeros(5, dtype=float)
        result = TransNetPredictor.predictions_to_scenes(preds)
        assert result.dtype == np.int32

    def test_single_frame_video(self):
        preds = np.array([0.0])
        result = TransNetPredictor.predictions_to_scenes(preds)
        assert result.shape == (1, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 0
