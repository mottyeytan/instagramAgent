"""Tests for encoder.py — face detection and encoding."""

import numpy as np
import tempfile
from PIL import Image

import encoder
from encoder import detect_face_locations, encode_faces, encode_primary_face


def _create_test_image(width=200, height=200, color=(128, 128, 128)) -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    return tmp.name


class TestEncodeFaces:
    def test_no_face_in_solid_image(self):
        """A solid color image should return no face embeddings."""
        path = _create_test_image(color=(100, 100, 100))
        result = encode_faces(path)
        assert result == []

    def test_nonexistent_file(self):
        """Non-existent file should return empty list, not crash."""
        result = encode_faces("/tmp/nonexistent_image_12345.jpg")
        assert result == []

    def test_corrupt_file(self):
        """A file that isn't a valid image should return empty list."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(b"not an image at all")
        tmp.close()
        result = encode_faces(tmp.name)
        assert result == []

    def test_embedding_shape(self):
        """If a face is found, each embedding should be a numpy array."""
        path = _create_test_image()
        result = encode_faces(path)
        assert result == []

    def test_encode_faces_uses_strict_detection(self, monkeypatch):
        path = _create_test_image()
        captured = {}

        def fake_represent(*args, **kwargs):
            captured.update(kwargs)
            return [{
                "embedding": [0.1, 0.2, 0.3],
                "facial_area": {"x": 10, "y": 20, "w": 30, "h": 40},
                "face_confidence": 0.99,
            }]

        monkeypatch.setattr(encoder.DeepFace, "represent", fake_represent)

        result = encode_faces(path)

        assert captured["enforce_detection"] is True
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float64

    def test_zero_confidence_full_frame_detection_is_filtered(self, monkeypatch):
        path = _create_test_image(width=150, height=150)

        def fake_represent(*args, **kwargs):
            return [{
                "embedding": [0.1, 0.2, 0.3],
                "facial_area": {"x": 0, "y": 0, "w": 149, "h": 149},
                "face_confidence": 0.0,
            }]

        monkeypatch.setattr(encoder.DeepFace, "represent", fake_represent)

        assert encode_faces(path) == []
        assert detect_face_locations(path) == []

    def test_encode_primary_face_keeps_single_best_face(self, monkeypatch):
        path = _create_test_image(width=200, height=200)

        def fake_represent(*args, **kwargs):
            return [
                {
                    "embedding": [1.0, 0.0, 0.0],
                    "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10},
                    "face_confidence": 0.99,
                },
                {
                    "embedding": [0.0, 1.0, 0.0],
                    "facial_area": {"x": 75, "y": 70, "w": 30, "h": 30},
                    "face_confidence": 0.99,
                },
                {
                    "embedding": [0.0, 0.0, 1.0],
                    "facial_area": {"x": 150, "y": 150, "w": 20, "h": 20},
                    "face_confidence": 0.99,
                },
            ]

        monkeypatch.setattr(encoder.DeepFace, "represent", fake_represent)

        result = encode_primary_face(path)

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], np.array([0.0, 1.0, 0.0], dtype=np.float64))
