from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

COCO_WEIGHTS = MODELS_DIR / "yolov8s.pt"
SHOPPING_WEIGHTS = MODELS_DIR / "shoppingbest5.pt"
TRAFFIC_LIGHT_WEIGHTS = MODELS_DIR / "trafficlight.pt"
# Same file covers crosswalk + tactile-paving segmentation; the crossing mode
# filters by class name so only crosswalk-like classes drive the alignment.
SEGMENTATION_WEIGHTS = MODELS_DIR / "yolo-seg.pt"
HAND_LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"

YOLO_CONF = 0.4
YOLO_IMGSZ = 640

WS_HOST = "0.0.0.0"
WS_PORT = 8765

STT_MODEL = "small"
STT_LANGUAGE = "zh"
# Force CPU. "auto" picks CUDA when torch is built with CUDA support, even on
# machines without the matching cuBLAS DLLs, which then crashes at first
# inference. CPU + int8 runs at ~0.5–1 s per 1.5 s chunk on modern laptops,
# which is fine for this demo. Switch to "cuda" only if you've verified the
# CUDA toolkit is installed and cublas64_12.dll resolves on PATH.
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"
STT_CHUNK_SECONDS = 1.5

TTS_VOICE = "zh-TW-HsiaoChenNeural"
TTS_RATE = "+0%"

TTS_THROTTLE_SEC = 1.5

VOICE_SEARCH_TIMEOUT_SEC = 30.0
VOICE_SEARCH_CENTER_LO = 0.35
VOICE_SEARCH_CENTER_HI = 0.65
VOICE_SEARCH_NEAR_AREA = 0.15
VOICE_SEARCH_FOUND_FRAMES = 5

CROSSING_NO_CROSSWALK_FRAMES = 15
CROSSING_DRIFT_TOLERANCE = 0.18

HAND_ALIGN_TOLERANCE = 0.08
HAND_ALIGN_HOLD_FRAMES = 5

CROSSING_CLASSES = {
    "crosswalk": ("crosswalk", "zebra_crossing", "zebra"),
    "red_light": ("red_light", "red", "traffic_light_red"),
    "green_light": ("green_light", "green", "traffic_light_green"),
}
