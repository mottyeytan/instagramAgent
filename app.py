"""Instagram Face Matcher — Streamlit UI."""

import subprocess
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from encoder import encode_faces, detect_face_locations, warmup
from matcher import find_matches, cosine_distance, Match
from scraper import scrape_account, get_cached_stats, DB_PATH, PHOTOS_DIR
from app_v4_helpers import (
    BACKEND_URL,
    check_backend_health,
    consume_sse_stream,
    format_event_display,
)

INPUT_DIR = Path("data/input")

st.set_page_config(page_title="Instagram Face Matcher", layout="wide")

# Warm up DeepFace model on first run
if "model_ready" not in st.session_state:
    with st.spinner("Loading face recognition model..."):
        warmup()
    st.session_state.model_ready = True


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


def render_v3_ui():
    """Render the classic V3 face-matching UI."""
    # Detect faces FIRST (cached, runs once)
    input_photos = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.jpeg")) + list(INPUT_DIR.glob("*.png"))

    if not input_photos:
        st.info("No photos found. Drop your photos into `data/input/` and refresh.")
        return

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

    target = st.text_input("Username to search", value="aardvarkisrael")

    batch_size = st.number_input("Batch size", min_value=10, max_value=500, value=50, step=10)

    if st.button("Scan Network", disabled=not target):
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
                target_username=target,
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

    # --- Backend control ---
    st.divider()
    st.header("V4 Backend")
    if st.button("Start Backend", key="start_backend"):
        subprocess.Popen(
            ["python", "-m", "uvicorn", "backend.server:app", "--port", "8000"]
        )
        st.success("Backend starting on port 8000...")


# ---------------------------------------------------------------------------
# V4 Agent UI
# ---------------------------------------------------------------------------


def _init_v4_session_state():
    """Ensure all V4 session-state keys exist."""
    if "v4_activity" not in st.session_state:
        st.session_state.v4_activity = []
    if "v4_chat" not in st.session_state:
        st.session_state.v4_chat = []
    if "v4_investigation_id" not in st.session_state:
        st.session_state.v4_investigation_id = None
    if "v4_waiting_interrupt" not in st.session_state:
        st.session_state.v4_waiting_interrupt = False
    if "v4_interrupt_event" not in st.session_state:
        st.session_state.v4_interrupt_event = None
    if "v4_complete" not in st.session_state:
        st.session_state.v4_complete = False


def _render_activity_event(event: dict):
    """Render a single event in the activity feed column."""
    etype = event.get("type", "unknown")

    if etype == "scanning":
        username = event.get("username", "?")
        platform = event.get("platform", "instagram")
        st.info(f"\U0001f50d Scanning @{username} on {platform}...")

    elif etype == "found_leads":
        count = event.get("count", 0)
        platform = event.get("platform", "instagram")
        st.success(f"\U0001f4cb Found {count} leads on {platform}")

    elif etype == "face_matched":
        username = event.get("username", "?")
        score = event.get("score", 0)
        photo = event.get("photo_path")
        cols = st.columns([1, 4])
        with cols[0]:
            if photo and Path(photo).exists():
                st.image(photo, width=48)
        with cols[1]:
            st.success(f"\u2705 MATCH: @{username} ({score}%)")

    elif etype == "face_rejected":
        username = event.get("username", "?")
        score = event.get("score", 0)
        st.markdown(
            f"<span style='opacity:0.45'>\u274c @{username} ({score}%)</span>",
            unsafe_allow_html=True,
        )

    elif etype == "budget_update":
        spent = event.get("spent", 0)
        total = event.get("total", 1)
        pct = min(spent / max(total, 1), 1.0)
        st.progress(pct, text=f"Budget: {spent}/{total}")

    elif etype == "investigation_complete":
        matches = event.get("matches_found", 0)
        st.balloons()
        st.success(f"\U0001f3c1 Investigation complete — {matches} matches found")

    else:
        st.write(format_event_display(event))


def _process_sse_events(events_iter):
    """Consume an SSE event iterator, updating session state for each event."""
    for event in events_iter:
        st.session_state.v4_activity.append(event)
        etype = event.get("type", "unknown")

        if etype == "interrupt":
            st.session_state.v4_waiting_interrupt = True
            st.session_state.v4_interrupt_event = event
            st.session_state.v4_chat.append(
                {"role": "assistant", "content": event.get("question", "Agent needs input")}
            )
            # Stop consuming — we need user input before continuing
            return

        if etype == "investigation_complete":
            st.session_state.v4_complete = True


def render_v4_ui():
    """Render the V4 agent-based investigation UI."""
    _init_v4_session_state()

    backend_ok = check_backend_health()
    if not backend_ok:
        st.warning(
            "V4 backend is not running. Click **Start Backend** in the sidebar, "
            "or run `uvicorn backend.server:app --port 8000` manually. "
            "Falling back to direct investigation mode."
        )

    # --- Layout: left = activity feed, right = chat ---
    col_activity, col_chat = st.columns([3, 2])

    # ----- LEFT: Activity Feed -----
    with col_activity:
        st.subheader("Activity Feed")

        # Upload photos
        uploaded_files = st.file_uploader(
            "Upload reference photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple=True,
            key="v4_upload",
        )

        target = st.text_input("Target username", value="", key="v4_target")

        if st.button("Start Investigation", type="primary", key="v4_start"):
            if not uploaded_files:
                st.error("Upload at least one reference photo.")
            elif not target:
                st.error("Enter a target username.")
            elif not backend_ok:
                st.error("Backend is not running. Start it from the sidebar first.")
            else:
                # Save uploaded files to data/input/
                INPUT_DIR.mkdir(parents=True, exist_ok=True)
                photo_paths = []
                for uf in uploaded_files:
                    dest = INPUT_DIR / uf.name
                    dest.write_bytes(uf.getbuffer())
                    photo_paths.append(str(dest))

                # Reset state
                st.session_state.v4_activity = []
                st.session_state.v4_chat = []
                st.session_state.v4_waiting_interrupt = False
                st.session_state.v4_interrupt_event = None
                st.session_state.v4_complete = False

                inv_id = str(uuid.uuid4())
                st.session_state.v4_investigation_id = inv_id

                start_url = f"{BACKEND_URL}/investigations/start"
                body = {
                    "investigation_id": inv_id,
                    "target_username": target,
                    "photo_paths": photo_paths,
                }

                try:
                    events = consume_sse_stream(start_url, json_body=body)
                    _process_sse_events(events)
                except Exception as exc:
                    st.error(f"Backend error: {exc}")

        # Render accumulated events
        for ev in st.session_state.v4_activity:
            _render_activity_event(ev)

    # ----- RIGHT: Chat Panel -----
    with col_chat:
        st.subheader("Chat")

        # Show chat history
        for msg in st.session_state.v4_chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        user_input = st.chat_input("Type a message...", key="v4_chat_input")
        if user_input:
            st.session_state.v4_chat.append({"role": "user", "content": user_input})

            if st.session_state.v4_waiting_interrupt and st.session_state.v4_investigation_id:
                # Resume the investigation
                inv_id = st.session_state.v4_investigation_id
                resume_url = f"{BACKEND_URL}/investigations/{inv_id}/resume"
                body = {"response": user_input}

                st.session_state.v4_waiting_interrupt = False
                st.session_state.v4_interrupt_event = None

                try:
                    events = consume_sse_stream(resume_url, json_body=body)
                    _process_sse_events(events)
                except Exception as exc:
                    st.error(f"Backend error: {exc}")

                st.rerun()
            else:
                # Regular chat message — just display it
                st.rerun()


# ---------------------------------------------------------------------------
# Tab layout — V3 (classic) + V4 (agent)
# ---------------------------------------------------------------------------
# --- UI ---

st.title("Instagram Face Matcher")

tab_v3, tab_v4 = st.tabs(["V3 Classic", "V4 Agent"])

with tab_v3:
    render_v3_ui()

with tab_v4:
    render_v4_ui()
