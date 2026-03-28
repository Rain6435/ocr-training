import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import json

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Historical OCR Pipeline Dashboard", layout="wide")
st.title("Historical Document OCR Pipeline")

# === SIDEBAR ===
st.sidebar.header("Configuration")
output_format = st.sidebar.selectbox("Output Format", ["text", "pdf", "tei-xml"])
force_engine = st.sidebar.selectbox(
    "Force Engine", [None, "tesseract", "custom", "trocr"],
    format_func=lambda x: "Auto (intelligent routing)" if x is None else x,
)

st.sidebar.subheader("Routing Thresholds")
easy_thresh = st.sidebar.slider("Easy Threshold", 0.0, 1.0, 0.7, 0.05)
hard_thresh = st.sidebar.slider("Hard Threshold", 0.0, 1.0, 0.6, 0.05)
escalation_thresh = st.sidebar.slider("Escalation Threshold", 0.0, 1.0, 0.5, 0.05)

if st.sidebar.button("Update Routing Config"):
    try:
        resp = requests.put(f"{API_BASE}/config/routing", json={
            "easy_threshold": easy_thresh,
            "hard_threshold": hard_thresh,
            "escalation_threshold": escalation_thresh,
        })
        if resp.ok:
            st.sidebar.success("Config updated!")
        else:
            st.sidebar.error(f"Failed: {resp.text}")
    except requests.ConnectionError:
        st.sidebar.error("API not reachable")

# === TABS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Inference", "Pipeline Stats", "Training Curves",
    "Benchmark Results", "Review Queue",
])

# === TAB 1: LIVE INFERENCE ===
with tab1:
    uploaded = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if uploaded:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Preprocessed")
            try:
                uploaded.seek(0)
                resp = requests.post(
                    f"{API_BASE}/preprocess",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                )
                if resp.ok:
                    preprocessed = Image.open(io.BytesIO(resp.content))
                    st.image(preprocessed, use_container_width=True)
                    skew = resp.headers.get("X-Skew-Angle", "N/A")
                    st.caption(f"Skew angle: {skew}°")
            except requests.ConnectionError:
                st.warning("API not reachable — showing original")

        # Run OCR
        if st.button("Run OCR"):
            uploaded.seek(0)
            params = {"output_format": output_format}
            if force_engine:
                params["force_engine"] = force_engine
            try:
                resp = requests.post(
                    f"{API_BASE}/ocr/single",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                    params=params,
                )
                if resp.ok:
                    result = resp.json()

                    # Engine badge
                    engine = result["engine_used"]
                    difficulty = result["difficulty"]
                    colors = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                    badge = colors.get(difficulty, "⚪")

                    st.subheader("OCR Result")
                    st.markdown(f"**Engine:** {badge} {engine} | **Difficulty:** {difficulty}")
                    st.markdown(f"**Confidence:** {result['confidence']:.1%} | "
                                f"**Time:** {result['processing_time_ms']:.0f}ms | "
                                f"**Cost:** ${result['cost']:.4f}")
                    if result["needs_review"]:
                        st.warning("⚠️ Flagged for human review (low confidence)")
                    st.text_area("Transcription", result["text"], height=200)
                    st.caption(f"Corrections applied: {result['corrections_applied']}")
                else:
                    st.error(f"OCR failed: {resp.text}")
            except requests.ConnectionError:
                st.error("API not reachable")

# === TAB 2: PIPELINE STATS ===
with tab2:
    st.subheader("Pipeline Statistics")
    try:
        resp = requests.get(f"{API_BASE}/stats")
        if resp.ok:
            stats = resp.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Processed", stats["total_processed"])
            col2.metric("Avg Confidence", f"{stats['average_confidence']:.1%}")
            col3.metric("Total Cost", f"${stats['total_cost']:.4f}")
            col4.metric("Avg Time", f"{stats['average_processing_time_ms']:.0f}ms")

            # Routing distribution
            st.subheader("Routing Distribution")
            routing = {
                "Easy (Tesseract)": stats["easy_count"],
                "Medium (Custom CRNN)": stats["medium_count"],
                "Hard (TrOCR)": stats["hard_count"],
            }
            st.bar_chart(routing)
            st.caption(f"Escalated: {stats['escalated_count']}")
    except requests.ConnectionError:
        st.info("Start the API to see live stats: `make run-api`")

# === TAB 3: TRAINING CURVES ===
with tab3:
    st.subheader("Training Curves (TensorBoard)")
    st.markdown("### Embedded TensorBoard")
    st.components.v1.iframe("http://localhost:6006", height=600)
    st.caption("Start TensorBoard: `make run-tensorboard`")

# === TAB 4: BENCHMARK RESULTS ===
with tab4:
    st.subheader("Benchmark Results")
    st.info("Run benchmarks with `make benchmark` to populate this section.")

    # Placeholder tables
    st.markdown("### CER Comparison")
    st.dataframe({
        "Engine": ["Tesseract", "Custom CRNN", "TrOCR-large", "PaddleOCR", "Intelligent Routing"],
        "IAM Lines": ["—"] * 5,
        "IAM Words": ["—"] * 5,
        "EMNIST": ["—"] * 5,
    })

    st.markdown("### Cost per 1000 Pages")
    st.dataframe({
        "Approach": ["All Tesseract", "All Custom CRNN", "All TrOCR", "Intelligent Routing"],
        "Cost": ["$0", "$1.00", "$50.00", "TBD"],
    })

# === TAB 5: REVIEW QUEUE ===
with tab5:
    st.subheader("Documents Flagged for Review")
    st.info("Low-confidence documents will appear here after processing.")
