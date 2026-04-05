"""Real InsightFace integration tests for encoder.py.

These tests use the actual InsightFace buffalo_l model (no mocks).
They verify the encoder works end-to-end without crashing.
"""

import tempfile

import numpy as np
import pytest
from PIL import Image

from encoder import (
    detect_face_locations,
    encode_faces,
    warmup,
)


def _create_solid_image(width=200, height=200, color=(128, 128, 128)) -> str:
    """Create a solid color test image and return its path."""
    img = Image.new("RGB", (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    return tmp.name


def _create_face_like_image() -> str:
    """Create an image with face-like features using simple drawing.

    This is a synthetic test image. InsightFace may or may not detect a
    face in it — the test just verifies the function doesn't crash and
    returns the correct types.
    """
    import cv2

    img = np.ones((400, 400, 3), dtype=np.uint8) * 200

    # Draw a skin-toned oval for a head
    cv2.ellipse(img, (200, 180), (80, 100), 0, 0, 360, (180, 150, 130), -1)
    # Eyes
    cv2.circle(img, (170, 160), 8, (40, 40, 40), -1)
    cv2.circle(img, (230, 160), 8, (40, 40, 40), -1)
    # Nose
    cv2.line(img, (200, 170), (200, 195), (140, 120, 100), 2)
    # Mouth
    cv2.ellipse(img, (200, 215), (25, 10), 0, 0, 180, (100, 80, 80), 2)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


class TestRealEncoder:
    """Integration tests using the real InsightFace model."""

    def test_real_warmup(self):
        """warmup() should load the model without crashing."""
        warmup()
        # If we get here, the model loaded successfully.

    def test_real_encode_no_face(self):
        """A solid color image should return an empty list (no faces)."""
        path = _create_solid_image(color=(100, 100, 100))
        result = encode_faces(path)
        assert isinstance(result, list)
        assert result == []

    def test_real_encode_returns_512_float32(self):
        """If a face is detected, embeddings should be 512-dim float32.

        Uses a face-like synthetic image. If InsightFace doesn't detect
        a face (expected for simple drawings), we just verify the return
        is an empty list. If it does detect one, we verify the shape and
        dtype.
        """
        path = _create_face_like_image()
        result = encode_faces(path)
        assert isinstance(result, list)

        # Either no face detected (acceptable) or valid embeddings
        for emb in result:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (512,), f"Expected 512-dim, got {emb.shape}"
            assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"

    def test_real_detect_locations(self):
        """detect_face_locations should return a list of dicts with x,y,w,h.

        Uses a face-like image. Whether or not a face is detected, the
        function should return a list and not crash.
        """
        path = _create_face_like_image()
        result = detect_face_locations(path)
        assert isinstance(result, list)

        for loc in result:
            assert isinstance(loc, dict)
            assert "x" in loc
            assert "y" in loc
            assert "w" in loc
            assert "h" in loc
            assert isinstance(loc["x"], int)
            assert isinstance(loc["y"], int)
            assert isinstance(loc["w"], int)
            assert isinstance(loc["h"], int)

    def test_real_nonexistent_file(self):
        """A missing file should return an empty list, not crash."""
        result = encode_faces("/tmp/nonexistent_image_9999999.jpg")
        assert isinstance(result, list)
        assert result == []
