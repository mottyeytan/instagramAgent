"""Tests for encoder.py V4 — InsightFace buffalo_l face detection and encoding."""

import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
from PIL import Image
import pytest


# ---------------------------------------------------------------------------
# Mock the insightface package so tests work without it installed.
# We create a fake module hierarchy: insightface, insightface.app
# ---------------------------------------------------------------------------

_mock_insightface = types.ModuleType("insightface")
_mock_insightface_app = types.ModuleType("insightface.app")

# FaceAnalysis will be a MagicMock that we can configure per-test
_MockFaceAnalysis = MagicMock()
_mock_insightface_app.FaceAnalysis = _MockFaceAnalysis
_mock_insightface.app = _mock_insightface_app

sys.modules.setdefault("insightface", _mock_insightface)
sys.modules.setdefault("insightface.app", _mock_insightface_app)


# ---------------------------------------------------------------------------
# Helper: create a fake Face object (SimpleNamespace works like InsightFace Face)
# ---------------------------------------------------------------------------

def _make_face(embedding: np.ndarray, bbox: list, det_score: float):
    """Return a mock face object matching InsightFace Face attributes."""
    face = types.SimpleNamespace()
    face.normed_embedding = embedding
    face.bbox = np.array(bbox, dtype=np.float32)
    face.det_score = det_score
    return face


def _make_normalized_embedding(dim=512, seed=42):
    """Return a random L2-normalized float32 embedding of given dimension."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _create_test_image(width=200, height=200, color=(128, 128, 128)) -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_encoder_singleton():
    """Reset the encoder module's lazy-init singleton before each test."""
    # Force re-import so module-level state is clean
    if "encoder" in sys.modules:
        del sys.modules["encoder"]
    _MockFaceAnalysis.reset_mock()
    yield
    # Cleanup after test
    if "encoder" in sys.modules:
        del sys.modules["encoder"]


@pytest.fixture
def mock_app():
    """Provide a pre-configured mock FaceAnalysis app instance."""
    app_instance = MagicMock()
    _MockFaceAnalysis.return_value = app_instance
    return app_instance


# ===========================================================================
# 1. test_model_load_buffalo_l
# ===========================================================================

class TestModelLoadBuffaloL:
    def test_model_load_buffalo_l(self, mock_app):
        """FaceAnalysis must be created with name='buffalo_l' and prepare() called."""
        mock_app.get.return_value = []

        import encoder
        encoder._get_app()  # trigger lazy init

        _MockFaceAnalysis.assert_called_once()
        call_kwargs = _MockFaceAnalysis.call_args
        assert call_kwargs[1].get("name") == "buffalo_l" or (
            call_kwargs[0] and call_kwargs[0][0] == "buffalo_l"
        ), "FaceAnalysis must be created with name='buffalo_l'"

        # Verify providers include CUDA and CPU
        providers = call_kwargs[1].get("providers", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
        assert "CUDAExecutionProvider" in providers
        assert "CPUExecutionProvider" in providers

        mock_app.prepare.assert_called_once()
        prep_kwargs = mock_app.prepare.call_args
        assert prep_kwargs[1].get("ctx_id") == 0
        assert prep_kwargs[1].get("det_size") == (640, 640)


# ===========================================================================
# 2. test_model_load_fallback_to_cpu
# ===========================================================================

class TestModelLoadFallbackCPU:
    def test_model_load_fallback_to_cpu(self):
        """If CUDA init fails, encoder should fallback to CPU-only."""
        call_count = 0

        def fa_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            providers = kwargs.get("providers", [])
            if "CUDAExecutionProvider" in providers and call_count == 1:
                raise RuntimeError("CUDA not available")
            mock_instance = MagicMock()
            mock_instance.get.return_value = []
            return mock_instance

        _MockFaceAnalysis.side_effect = fa_side_effect

        import encoder
        app = encoder._get_app()

        assert call_count == 2, "Should have tried twice: once with CUDA, once CPU-only"
        assert app is not None

        # Cleanup side_effect
        _MockFaceAnalysis.side_effect = None


# ===========================================================================
# 3. test_warmup_calls_model
# ===========================================================================

class TestWarmup:
    def test_warmup_calls_model(self, mock_app):
        """warmup() should trigger a dummy inference via app.get()."""
        mock_app.get.return_value = []

        import encoder
        encoder.warmup()

        mock_app.get.assert_called()


# ===========================================================================
# 4. test_encode_faces_returns_512_dim_float32
# ===========================================================================

class TestEncodeFaces512:
    def test_encode_faces_returns_512_dim_float32(self, mock_app):
        """encode_faces() should return list of 512-dim float32 embeddings."""
        emb1 = _make_normalized_embedding(512, seed=1)
        emb2 = _make_normalized_embedding(512, seed=2)
        mock_app.get.return_value = [
            _make_face(emb1, [10, 20, 60, 80], 0.95),
            _make_face(emb2, [100, 100, 200, 200], 0.88),
        ]

        path = _create_test_image()

        import encoder
        result = encoder.encode_faces(path)

        assert len(result) == 2
        for emb in result:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (512,)
            assert emb.dtype == np.float32


# ===========================================================================
# 5. test_encode_faces_l2_normalized
# ===========================================================================

class TestEncodeFacesL2:
    def test_encode_faces_l2_normalized(self, mock_app):
        """Each embedding should be L2-normalized (norm ~= 1.0)."""
        emb = _make_normalized_embedding(512, seed=7)
        mock_app.get.return_value = [
            _make_face(emb, [10, 20, 60, 80], 0.95),
        ]

        path = _create_test_image()

        import encoder
        result = encoder.encode_faces(path)

        assert len(result) == 1
        assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-5


# ===========================================================================
# 6. test_encode_faces_empty_on_no_face
# ===========================================================================

class TestEncodeFacesEmpty:
    def test_encode_faces_empty_on_no_face(self, mock_app):
        """A solid color image with no faces detected should return []."""
        mock_app.get.return_value = []

        path = _create_test_image(color=(50, 50, 50))

        import encoder
        result = encoder.encode_faces(path)

        assert result == []


# ===========================================================================
# 7. test_encode_primary_face_selects_largest_centered
# ===========================================================================

class TestEncodePrimaryFace:
    def test_encode_primary_face_selects_largest_centered(self, mock_app):
        """Primary face = max(area * center_proximity). The big centered face wins."""
        # Image is 200x200, center at (100, 100)
        # Face A: small, top-left corner — area=100, far from center
        emb_a = _make_normalized_embedding(512, seed=10)
        face_a = _make_face(emb_a, [0, 0, 10, 10], 0.99)

        # Face B: large, near center — area=2500, close to center (WINNER)
        emb_b = _make_normalized_embedding(512, seed=20)
        face_b = _make_face(emb_b, [75, 75, 125, 125], 0.98)

        # Face C: medium, bottom-right corner — area=400, far from center
        emb_c = _make_normalized_embedding(512, seed=30)
        face_c = _make_face(emb_c, [160, 160, 180, 180], 0.97)

        mock_app.get.return_value = [face_a, face_b, face_c]

        path = _create_test_image(width=200, height=200)

        import encoder
        result = encoder.encode_primary_face(path)

        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], emb_b)


# ===========================================================================
# 8. test_zero_det_score_filtered
# ===========================================================================

class TestZeroDetScoreFiltered:
    def test_zero_det_score_filtered(self, mock_app):
        """Faces with det_score <= 0 should be excluded."""
        good_emb = _make_normalized_embedding(512, seed=40)
        bad_emb = _make_normalized_embedding(512, seed=41)

        mock_app.get.return_value = [
            _make_face(good_emb, [10, 20, 60, 80], 0.9),
            _make_face(bad_emb, [100, 100, 200, 200], 0.0),   # det_score=0 -> filtered
            _make_face(bad_emb, [100, 100, 200, 200], -0.1),  # det_score<0 -> filtered
        ]

        path = _create_test_image()

        import encoder
        result = encoder.encode_faces(path)

        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], good_emb)


# ===========================================================================
# 9. test_nonexistent_file_returns_empty
# ===========================================================================

class TestNonexistentFile:
    def test_nonexistent_file_returns_empty(self, mock_app):
        """Missing file should return [] without crashing."""
        import encoder
        result = encoder.encode_faces("/tmp/absolutely_no_such_file_xyz_999.jpg")
        assert result == []


# ===========================================================================
# 10. test_detect_face_locations_format
# ===========================================================================

class TestDetectFaceLocations:
    def test_detect_face_locations_format(self, mock_app):
        """detect_face_locations should return dicts with x,y,w,h from bbox."""
        emb = _make_normalized_embedding(512, seed=50)
        # bbox = [x1, y1, x2, y2] = [10, 20, 60, 80]
        # expected: x=10, y=20, w=50, h=60
        mock_app.get.return_value = [
            _make_face(emb, [10, 20, 60, 80], 0.95),
        ]

        path = _create_test_image()

        import encoder
        result = encoder.detect_face_locations(path)

        assert len(result) == 1
        loc = result[0]
        assert set(loc.keys()) == {"x", "y", "w", "h"}
        assert loc["x"] == 10
        assert loc["y"] == 20
        assert loc["w"] == 50
        assert loc["h"] == 60
