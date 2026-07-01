from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Streamlit is not installed. Run: pip install -r requirements.txt") from exc

from src.constants import DISCLAIMER
from src.inference import screen_retina_image

st.set_page_config(page_title="RetinaAI Screening", layout="wide")
st.title("RetinaAI Screening")
st.caption("Uncertainty-aware explainable retinal screening prototype")
st.warning(DISCLAIMER)

uploaded = st.file_uploader("Upload retinal fundus image", type=["png", "jpg", "jpeg", "tif", "tiff"])
model_path = st.sidebar.text_input("Model path", "models/baseline_sklearn.pkl")
thresholds_path = st.sidebar.text_input("Thresholds path", "configs/thresholds.yaml")

if uploaded:
    suffix = Path(uploaded.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getbuffer())
        image_path = Path(handle.name)

    result = screen_retina_image(image_path, model_path=model_path, thresholds_path=thresholds_path)
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Image")
        st.image(str(image_path), use_container_width=True)
        explanation_path = result["outputs"].get("explanation_path")
        if explanation_path:
            st.subheader("Explanation")
            st.image(explanation_path, use_container_width=True)

    with right:
        st.subheader("Screening Result")
        st.json(result["prediction"])
        st.subheader("Quality Gate")
        st.json(result["quality"])
        st.subheader("Uncertainty Routing")
        st.json(result["uncertainty"])
        report_path = result["outputs"].get("report_path")
        if report_path and Path(report_path).exists():
            st.download_button(
                "Download AI Screening Report",
                data=Path(report_path).read_bytes(),
                file_name=Path(report_path).name,
                mime="application/pdf",
            )

    with st.expander("Raw result JSON"):
        st.code(json.dumps(result, indent=2), language="json")
else:
    st.info("Upload an image to run quality checks, model inference if weights exist, uncertainty routing, and report generation.")
