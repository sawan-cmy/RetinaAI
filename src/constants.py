from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DR_CLASSES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}
DISCLAIMER = (
    "This AI output is a screening prototype and is not a medical diagnosis. "
    "Clinical diagnosis requires review by a qualified medical professional."
)
