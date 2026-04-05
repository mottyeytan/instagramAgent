"""Unit tests for encoder.py — mock-based, no real InsightFace model needed.

Uses monkeypatch to replace the singleton app so tests are fast and
don't require the buffalo_l model to be downloaded.
"""

import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import encoder
from encoder import (
    detect_face_locations,
    encode_faces,
    encode_primary_face,
)


def _create_test_image(width=200, height=200, color=(128, 128, 128)) -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    return tmp.name


def _make_fake_face(embedding, bbox, det_score=0.99):
    """Create a fake face object mimicking InsightFace's Face namedtuple."""
    face = SimpleNamespace()
    face.embedding = np.array(embedding, dtype=np.float32)
    face.bbox = np.array(bbox, dtype=np.float32)
    face.det_score = det_score
    return face


class TestEncodeFacesUnit:
    """Unit tests using mocked InsightFace model."""

    def test_no_face_in_solid_image(self, monkeypatch):
        """When the model returns no faces, encode_faces returns []."""
        path = _create_test_image(color=(100, 100, 100))

        fake_app = SimpleNamespace()
        fake_app.get = lambda img: []
        monkeypatch.setattr(encoder, "_app", fake_app)

        result = encode_faces(path)
        assert result == []

    def test_nonexistent_file(self, monkeypatch):
        """Non-existent file should return empty list, not crash."""
        fake_app = SimpleNamespace()
        fake_app.get = lambda img: []
        monkeypatch.setattr(encoder, "_app", fake_app)

        result = encode_faces("/tmp/nonexistent_image_12345.jpg")
        assert result == []

    def test_corrupt_file(self, monkeypatch):
        """A file that isn't a valid image should return empty list."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(b"not an image at all")
        tmp.close()

        fake_app = SimpleNamespace()
        fake_app.get = lambda img: []
        monkeypatch.setattr(encoder, "_app", fake_app)

        result = encode_faces(tmp.name)
        assert result == []

    def test_embedding_shape_and_dtype(self, monkeypatch):
        """Embeddings should be 512-dim float32 numpy arrays."""
        path = _create_test_image()
        emb_512 = np.random.randn(512).astype(np.float32)

        face = _make_fake_face(
            embedding=emb_512,
            bbox=[10, 20, 40, 60],
        )
        fake_app = SimpleNamespace()
        fake_app.get = lambda img: [face]
        monkeypatch.setattr(encoder, "_app", fake_app)

        result = encode_faces(path)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32
        assert result[0].shape == (512,)

    def test_encode_primary_face_picks_largest_centered(self, monkeypatch):
        """Primary face should be the largest, most centered face."""
        path = _create_test_image(width=200, height=200)

        small_corner = _make_fake_face(
            embedding=np.array([1.0] + [0.0] * 511, dtype=np.float32),
            bbox=[0, 0, 10, 10],
        )
        big_center = _make_fake_face(
            embedding=np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32),
            bbox=[70, 65, 130, 135],
        )
        small_far = _make_fake_face(
            embedding=np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32),
            bbox=[150, 150, 170, 170],
        )

        fake_app = SimpleNamespace()
        fake_app.get = lambda img: [small_corner, big_center, small_far]
        monkeypatch.setattr(encoder, "_app", fake_app)

        result = encode_primary_face(path)
        assert len(result) == 1
        # Should pick big_center (largest area, most centered)
        np.testing.assert_array_almost_equal(
            result[0][:2],
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_detect_face_locations_returns_xywh(self, monkeypatch):
        """detect_face_locations should return dicts with x, y, w, h."""
        path = _create_test_image()

        face = _make_fake_face(
            embedding=np.random.randn(512).astype(np.float32),
            bbox=[10, 20, 50, 80],
        )
        fake_app = SimpleNamespace()
        fake_app.get = lambda img: [face]
        monkeypatch.setattr(encoder, "_app", fake_app)

        result = detect_face_locations(path)
        assert len(result) == 1
        loc = result[0]
        assert loc["x"] == 10
        assert loc["y"] == 20
        assert loc["w"] == 40  # 50 - 10
        assert loc["h"] == 60  # 80 - 20

    def test_encode_primary_face_no_faces(self, monkeypatch):
        """encode_primary_face with no faces should return []."""
        path = _create_test_image()

        fake_app = SimpleNamespace()
        fake_app.get = lambda img: []
        monkeypatch.setattr(encoder, "_app", fake_app)

        result = encode_primary_face(path)
        assert result == []
