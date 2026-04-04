"""Face detection and encoding using DeepFace (ArcFace model)."""

from pathlib import Path

import numpy as np
from deepface import DeepFace
from PIL import Image


MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"


def warmup():
    """Pre-load the model so first real call isn't slow."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    tmp_path = "/tmp/_deepface_warmup.jpg"
    Image.fromarray(dummy).save(tmp_path)
    try:
        DeepFace.represent(
            tmp_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
    except Exception:
        pass


def _represent_faces(image_path: str) -> list[dict]:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return []

    try:
        results = DeepFace.represent(
            img_path=str(path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except Exception:
        return []

    if isinstance(results, dict):
        results = [results]

    faces = []
    for face in results:
        embedding = face.get("embedding")
        region = face.get("facial_area", {})
        face_confidence = face.get("face_confidence")
        width = int(region.get("w", 0) or 0)
        height = int(region.get("h", 0) or 0)

        if not embedding or width <= 0 or height <= 0:
            continue
        if face_confidence is not None and face_confidence <= 0:
            continue

        faces.append(face)

    return faces


def _get_image_size(image_path: str) -> tuple[int, int] | None:
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


def _primary_face_key(face: dict, image_size: tuple[int, int] | None) -> tuple[float, float]:
    region = face.get("facial_area", {})
    x = float(region.get("x", 0) or 0)
    y = float(region.get("y", 0) or 0)
    w = float(region.get("w", 0) or 0)
    h = float(region.get("h", 0) or 0)
    area = w * h

    if image_size is None:
        return area, 0.0

    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return area, 0.0

    face_cx = x + (w / 2.0)
    face_cy = y + (h / 2.0)
    center_dx = (face_cx - (img_w / 2.0)) / max(img_w / 2.0, 1.0)
    center_dy = (face_cy - (img_h / 2.0)) / max(img_h / 2.0, 1.0)
    center_distance = float(np.hypot(center_dx, center_dy))
    return area, -center_distance


def encode_faces(image_path: str) -> list[np.ndarray]:
    """Detect and encode all faces in an image."""
    embeddings = []
    for face in _represent_faces(image_path):
        embedding = np.array(face["embedding"], dtype=np.float64)
        if embedding.shape[0] > 0:
            embeddings.append(embedding)
    return embeddings


def encode_primary_face(image_path: str) -> list[np.ndarray]:
    """Detect and encode only the most likely primary face in an image."""
    faces = _represent_faces(image_path)
    if not faces:
        return []

    image_size = _get_image_size(image_path)
    primary_face = max(faces, key=lambda face: _primary_face_key(face, image_size))
    embedding = np.array(primary_face["embedding"], dtype=np.float64)
    if embedding.shape[0] == 0:
        return []
    return [embedding]


def detect_face_locations(image_path: str) -> list[dict]:
    """Detect face bounding boxes in an image."""
    locations = []
    for face in _represent_faces(image_path):
        region = face.get("facial_area", {})
        locations.append({
            "x": region.get("x", 0),
            "y": region.get("y", 0),
            "w": region.get("w", 0),
            "h": region.get("h", 0),
        })

    return locations
