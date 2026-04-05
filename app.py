"""Instagram Face Matcher — Streamlit UI (V4 + Legacy)."""

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

from encoder import encode_faces, detect_face_locations, warmup
from matcher import find_matches, cosine_distance, Match
from scraper import scrape_account, get_cached_stats, DB_PATH, PHOTOS_DIR

from app_v4_helpers import (
    init_session_state,
    set_investigation_running,
    append_activity_event,
    append_message,
    save_uploaded_photos,
    parse_sse_event,
    format_event_display,
    check_backend_health,
)

INPUT_DIR = Path("data/input")
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Instagram Face Matcher", layout="wide")

# Warm up DeepFace model on first run
if "model_ready" not in st.session_state:
    with st.spinner("Loading face recognition model..."):
        warmup()
    st.session_state.model_ready = True

# Initialize V4 session state
init_session_state(st.session_state)


# ── Shared helpers (used by legacy UI) ──────────────────────────────────


def draw_face_boxes(image: Image.Image, locations: list[dict]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for loc in locations:
        x, y, w, h = loc["x"], loc["y"], loc["w"], loc["h"]
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
    return img


def render_match_card(match: Match):
    col1, col2 = st.columns([1, 3])
    with col1:
        photo = Path(match.photo_path)
        if photo.exists():
            st.image(str(photo), width=80)
        else:
            st.write("No photo")
    with col2:
        badge = {"follower": "👤", "following": "➡️", "mutual": "🤝"}.get(match.relationship, "")
        st.markdown(f"**@{match.username}** {badge}")
        if match.full_name:
            st.caption(match.full_name)
        st.progress(match.confidence / 100, text=f"{match.confidence:.0f}% confidence")


@st.cache_data
def detect_input_faces():
    """Detect and deduplicate faces from data/input/ — cached so it only runs once."""
    if not INPUT_DIR.exists():
        return [], []

    all_faces = []
    for img_path in sorted(INPUT_DIR.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        embeddings = encode_faces(str(img_path))
        for idx, emb in enumerate(embeddings):
            all_faces.append({
                "embedding": emb,
                "source_file": img_path.name,
                "face_index": idx,
                "image_path": str(img_path),
            })

    # Deduplicate faces
    if not all_faces:
        return [], []

    groups = []
    for face in all_faces:
        matched_group = None
        for group in groups:
            dist = cosine_distance(face["embedding"], group[0]["embedding"])
            if dist < 0.3:
                matched_group = group
                break
        if matched_group is not None:
            matched_group.append(face)
        else:
            groups.append([face])

    unique_faces = []
    for i, group in enumerate(groups):
        if len(group) == 1:
            rep = group[0].copy()
        else:
            avg_emb = np.mean([f["embedding"] for f in group], axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            rep = group[0].copy()
            rep["embedding"] = avg_emb
        rep["person_id"] = i + 1
        rep["photo_count"] = len(group)
        rep["source_files"] = list(set(f["source_file"] for f in group))
        unique_faces.append(rep)

    return all_faces, unique_faces


# ── V4 Investigation UI ────────────────────────────────────────────────


def start_investigation_ui(target, seed_username, uploaded_photos, time_limit):
    """Kick off an investigation in standalone or connected mode."""
    # Save uploaded photos
    if uploaded_photos:
        saved = save_uploaded_photos(uploaded_photos, target_dir=INPUT_DIR)
        append_activity_event(
            st.session_state.activity_log,
            {"type": "info", "text": f"Saved {len(saved)} photo(s) to {INPUT_DIR}"},
        )

    # Check backend connectivity
    health = check_backend_health(BACKEND_URL)

    if health["healthy"]:
        # Connected mode — stream from FastAPI backend
        set_investigation_running(st.session_state, True, investigation_id="connected")
        append_activity_event(
            st.session_state.activity_log,
            {"type": "info", "text": "Connected to backend. Starting investigation via API..."},
        )
        st.info("Backend connected. SSE streaming will appear here once the backend API is implemented.")
    else:
        # Standalone mode — run directly
        set_investigation_running(st.session_state, True, investigation_id="standalone")
        append_activity_event(
            st.session_state.activity_log,
            {"type": "info", "text": f"Backend not available ({health.get('error', 'unknown')}). Running standalone."},
        )
        append_activity_event(
            st.session_state.activity_log,
            {"type": "progress", "text": f"Standalone investigation: target={target}, seed=@{seed_username}, limit={time_limit}m"},
        )

        # In standalone mode, fall back to the existing scrape + match pipeline
        if seed_username:
            username = seed_username.lstrip("@")
            try:
                result = scrape_account(target_username=username, batch_size=50)
                append_activity_event(
                    st.session_state.activity_log,
                    {
                        "type": "done",
                        "text": (
                            f"Scan complete: {result['followers_scraped']} followers, "
                            f"{result['following_scraped']} following, "
                            f"{result['faces_found']} faces."
                        ),
                    },
                )
            except Exception as exc:
                append_activity_event(
                    st.session_state.activity_log,
                    {"type": "error", "text": f"Scan failed: {exc}"},
                )
        else:
            append_activity_event(
                st.session_state.activity_log,
                {"type": "error", "text": "No seed username provided."},
            )

        set_investigation_running(st.session_state, False)


def render_v4_ui():
    """Render the V4 split-screen investigation UI."""
    st.title("AI Investigation Agent")

    # Sidebar: investigation controls
    with st.sidebar:
        st.header("New Investigation")
        target = st.text_input(
            "Target description",
            placeholder="e.g., Find people connected to @username",
        )
        seed_username = st.text_input(
            "Seed Instagram username",
            placeholder="@username",
        )

        uploaded_photos = st.file_uploader(
            "Upload target photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        time_limit = st.slider("Time limit (minutes)", 1, 30, 10)

        if st.button("Start Investigation", type="primary"):
            start_investigation_ui(target, seed_username, uploaded_photos, time_limit)

    # Main area: split-screen
    col_activity, col_chat = st.columns([1, 1])

    with col_activity:
        st.subheader("Agent Activity")
        activity_container = st.container(height=500)
        with activity_container:
            if not st.session_state.activity_log:
                st.caption("No activity yet. Start an investigation from the sidebar.")
            else:
                for event in st.session_state.activity_log:
                    display = format_event_display(event)
                    event_type = event.get("type", "info")
                    if event_type == "error":
                        st.error(display)
                    elif event_type == "match":
                        st.success(display)
                    elif event_type == "done":
                        st.info(display)
                    else:
                        st.write(display)

    with col_chat:
        st.subheader("Chat")
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        user_input = st.chat_input("Talk to the agents...")
        if user_input:
            append_message(st.session_state.messages, "user", user_input)
            # Echo a placeholder response (real agent chat comes with full backend)
            append_message(
                st.session_state.messages,
                "assistant",
                "Agent chat is not yet connected. Investigation results appear in the Activity panel.",
            )
            st.rerun()


# ── Legacy Scanner UI ──────────────────────────────────────────────────


def render_legacy_ui():
    """The original V1 scanner interface."""
    st.title("Instagram Face Matcher")

    # Detect faces FIRST (cached, runs once)
    input_photos = (
        list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.jpeg"))
        + list(INPUT_DIR.glob("*.png"))
    )

    if not input_photos:
        st.info("No photos found. Drop your photos into `data/input/` and refresh.")
        unique_faces = []
    else:
        all_faces, unique_faces = detect_input_faces()

        # Show photos in a compact row
        st.subheader(f"{len(input_photos)} photos — {len(unique_faces)} unique people detected")
        cols = st.columns(min(len(input_photos), 6))
        for i, photo_path in enumerate(sorted(input_photos)):
            with cols[i % len(cols)]:
                img = Image.open(photo_path)
                locations = detect_face_locations(str(photo_path))
                if locations:
                    img = draw_face_boxes(img, locations)
                st.image(img, caption=f"{len(locations)} face(s)", width="stretch")

        if unique_faces:
            for face in unique_faces:
                st.caption(f"Person {face['person_id']}: seen in {', '.join(face['source_files'])}")

        st.divider()

        # FIND MATCHES — always visible
        stats = get_cached_stats()
        cache_incompatible = bool(stats and not stats.get("compatible", True))
        if stats:
            if cache_incompatible:
                st.error(
                    "Cached profile data was built with an older face pipeline. "
                    "Delete `data/faces.db` and `data/photos/`, then scan again."
                )
            else:
                st.success(f"**{stats['total']}** profiles cached ({stats['with_face']} with faces)")

        if st.button("Find Matches", type="primary", disabled=not unique_faces or cache_incompatible):
            if not Path(str(DB_PATH)).exists() or not stats:
                st.warning("No cached data yet. Start a scan in the sidebar first.")
            elif cache_incompatible:
                st.warning("Cached data must be rebuilt before matching can run.")
            else:
                embeddings = [f["embedding"] for f in unique_faces]
                matches = find_matches(embeddings, str(DB_PATH))

                for face, face_matches in zip(unique_faces, matches):
                    st.divider()
                    st.subheader(f"Person {face['person_id']}")
                    st.caption(f"Seen in: {', '.join(face['source_files'])}")

                    col_photo, col_matches = st.columns([1, 2])
                    with col_photo:
                        src_img = Image.open(face["image_path"])
                        src_locations = detect_face_locations(face["image_path"])
                        if src_locations:
                            src_img = draw_face_boxes(src_img, src_locations)
                        st.image(src_img, width="stretch")

                    with col_matches:
                        if face_matches:
                            for m in face_matches:
                                render_match_card(m)
                                st.write("")
                        else:
                            st.info("No confident matches found.")

    # Sidebar: scan controls (non-blocking for the main content)
    with st.sidebar:
        st.header("Scan Instagram")

        scan_target = st.text_input("Username to search", value="aardvarkisrael", key="legacy_scan_target")

        batch_size = st.number_input("Batch size", min_value=10, max_value=500, value=50, step=10)

        if st.button("Scan Network", disabled=not scan_target):
            progress_bar = st.progress(0, text="Starting scan...")
            status_text = st.empty()
            batch_text = st.empty()

            def on_progress(current, total, username):
                pct = current / max(total, 1)
                progress_bar.progress(pct, text=f"Scanning {current}/{total}")
                status_text.caption(f"@{username}")

            def on_batch(batch_stats):
                total_s = batch_stats["followers_scraped"] + batch_stats["following_scraped"]
                batch_text.success(f"{total_s} scanned, {batch_stats['faces_found']} faces. Click 'Find Matches' now!")

            try:
                result = scrape_account(
                    target_username=scan_target,
                    batch_size=batch_size,
                    progress_callback=on_progress,
                    batch_callback=on_batch,
                )
                st.success(
                    f"Done! {result['followers_scraped']} followers, "
                    f"{result['following_scraped']} following. "
                    f"{result['faces_found']} faces found."
                )
                if result.get("last_error"):
                    st.warning(f"Scan ended early: {result['last_error']}")
            except (ValueError, ConnectionError) as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Scan failed: {type(e).__name__}: {e}")

        cached = get_cached_stats()
        if cached:
            if cached.get("compatible", True):
                st.caption(f"Cached: {cached['total']} profiles | {cached['with_face']} faces")
            else:
                st.caption(f"Cached: {cached['total']} profiles | reset required")


# ── Main: tab layout ───────────────────────────────────────────────────

tab1, tab2 = st.tabs(["V4 Investigation", "Legacy Scanner"])

with tab1:
    render_v4_ui()

with tab2:
    render_legacy_ui()
