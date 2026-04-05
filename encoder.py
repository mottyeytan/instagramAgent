"""Face detection and encoding using InsightFace (buffalo_l model)."""

from pathlib import Path

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_app = None


def _get_app() -> FaceAnalysis:
    """Return the singleton FaceAnalysis instance, creating it on first call."""
    global _app
    if _app is not None:
        return _app

    try:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        # Fallback: CPU-only if CUDA is unavailable
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

    _app = app
    return _app


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_image(image_path: str) -> np.ndarray | None:
    """Load an image as a BGR numpy array (InsightFace expects BGR)."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
        # Convert RGB -> BGR for InsightFace
        return arr[:, :, ::-1].copy()
    except Exception:
        return None


def _get_image_size(image_path: str) -> tuple[int, int] | None:
    """Return (width, height) of the image, or None on error."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


def _filter_faces(faces: list) -> list:
    """Filter out faces with det_score <= 0 or zero-area bounding box."""
    filtered = []
    for face in faces:
        if face.det_score <= 0:
            continue
        x1, y1, x2, y2 = face.bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        filtered.append(face)
    return filtered


def _detect_faces(image_path: str) -> list:
    """Run face detection + recognition on image, return filtered Face objects."""
    img = _load_image(image_path)
    if img is None:
        return []
    try:
        app = _get_app()
        faces = app.get(img)
    except Exception:
        return []
    return _filter_faces(faces)


def _primary_face_key(face, image_size: tuple[int, int] | None) -> float:
    """Score a face by area * center_proximity for primary selection."""
    x1, y1, x2, y2 = face.bbox
    w = float(x2 - x1)
    h = float(y2 - y1)
    area = w * h

    if image_size is None or image_size[0] <= 0 or image_size[1] <= 0:
        return area

    img_w, img_h = image_size
    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0
    center_dx = (face_cx - (img_w / 2.0)) / max(img_w / 2.0, 1.0)
    center_dy = (face_cy - (img_h / 2.0)) / max(img_h / 2.0, 1.0)
    center_distance = float(np.hypot(center_dx, center_dy))

    # proximity = 1 when perfectly centered, approaches 0 far away
    center_proximity = max(1.0 - center_distance, 0.01)
    return area * center_proximity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def warmup():
    """Pre-load the model so first real call isn't slow."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    app = _get_app()
    try:
        app.get(dummy)
    except Exception:
        pass


def encode_faces(image_path: str) -> list[np.ndarray]:
    """Detect and encode all faces in an image.

    Returns list of 512-dim float32 L2-normalized embeddings.
    """
    faces = _detect_faces(image_path)
    embeddings = []
    for face in faces:
        emb = np.array(face.normed_embedding, dtype=np.float32)
        if emb.shape[0] > 0:
            embeddings.append(emb)
    return embeddings


def encode_primary_face(image_path: str) -> list[np.ndarray]:
    """Detect and encode only the most likely primary face in an image.

    Returns a list with one 512-dim float32 embedding, or [] if no face found.
    """
    faces = _detect_faces(image_path)
    if not faces:
        return []

    image_size = _get_image_size(image_path)
    primary = max(faces, key=lambda f: _primary_face_key(f, image_size))
    emb = np.array(primary.normed_embedding, dtype=np.float32)
    if emb.shape[0] == 0:
        return []
    return [emb]


def detect_face_locations(image_path: str) -> list[dict]:
    """Detect face bounding boxes in an image.

    Returns list of dicts with x, y, w, h keys (converted from bbox format).
    """
    faces = _detect_faces(image_path)
    locations = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        locations.append({
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
        })
    return locations
