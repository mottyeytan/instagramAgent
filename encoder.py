"""Face detection and encoding using InsightFace (buffalo_l model)."""

from pathlib import Path

import cv2
import numpy as np

_app = None


def _get_app():
    """Lazy singleton — load InsightFace model on first use."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        _app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def warmup():
    """Pre-load the model so first real call isn't slow."""
    _get_app()


def _read_image(image_path: str) -> np.ndarray | None:
    """Read an image from disk as BGR numpy array (what InsightFace expects)."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return img
    except Exception:
        return None


def _get_faces(image_path: str) -> list:
    """Detect all faces in an image and return InsightFace Face objects."""
    img = _read_image(image_path)
    if img is None:
        return []
    try:
        return _get_app().get(img)
    except Exception:
        return []


def _get_image_size(image_path: str) -> tuple[int, int] | None:
    """Return (width, height) or None."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


def _primary_face_key(face, image_size: tuple[int, int] | None) -> tuple[float, float]:
    """Score a face for primary selection: largest + most centered wins."""
    bbox = face.bbox  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w = x2 - x1
    h = y2 - y1
    area = w * h

    if image_size is None:
        return area, 0.0

    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return area, 0.0

    face_cx = x1 + (w / 2.0)
    face_cy = y1 + (h / 2.0)
    center_dx = (face_cx - (img_w / 2.0)) / max(img_w / 2.0, 1.0)
    center_dy = (face_cy - (img_h / 2.0)) / max(img_h / 2.0, 1.0)
    center_distance = float(np.hypot(center_dx, center_dy))
    return area, -center_distance


def encode_faces(image_path: str) -> list[np.ndarray]:
    """Detect and encode all faces in an image.

    Returns a list of 512-dimensional float32 embedding vectors.
    """
    faces = _get_faces(image_path)
    embeddings = []
    for face in faces:
        emb = face.embedding
        if emb is not None and emb.shape[0] > 0:
            embeddings.append(emb.astype(np.float32))
    return embeddings


def encode_primary_face(image_path: str) -> list[np.ndarray]:
    """Detect and encode only the most likely primary face in an image."""
    faces = _get_faces(image_path)
    if not faces:
        return []

    image_size = _get_image_size(image_path)
    primary = max(faces, key=lambda f: _primary_face_key(f, image_size))
    emb = primary.embedding
    if emb is None or emb.shape[0] == 0:
        return []
    return [emb.astype(np.float32)]


def detect_face_locations(image_path: str) -> list[dict]:
    """Detect face bounding boxes in an image.

    Returns list of dicts with x, y, w, h keys.
    """
    faces = _get_faces(image_path)
    locations = []
    for face in faces:
        bbox = face.bbox  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        locations.append({
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
        })
    return locations
