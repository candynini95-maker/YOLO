import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
from .config import ModelConfig

@dataclass
class DetectedObject:
    cx: int
    cy: int
    w: int
    h: int
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float
    cls_id: int
    cls_name: str

class YOLODetector:
    """
    YOLO Model detector wrapper supporting .pt and .onnx formats,
    with automatic GPU (CUDA) and CPU device selection.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model_path = self._resolve_model_path(config)
        self.model = None

        self._load_model()

    def _resolve_device(self, requested: str) -> str:
        """Resolve the optimal execution device (CUDA GPU vs CPU)."""
        if requested == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 [Detector] NVIDIA GPU detected: {gpu_name}. Using CUDA acceleration.")
                return "0"
            else:
                print("💻 [Detector] No CUDA GPU available in current Torch environment. Using CPU.")
                return "cpu"
        elif "cuda" in requested:
            if not torch.cuda.is_available():
                print("⚠️ [Detector] CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            return requested.replace("cuda:", "") if ":" in requested else "0"
        return requested

    def _resolve_model_path(self, config: ModelConfig) -> str:
        """Select existing model weights (.pt or .onnx)."""
        # If running on CPU or if .pt does not exist, check if ONNX is available
        if os.path.exists(config.weights):
            return config.weights
        if os.path.exists(config.onnx_weights):
            print(f"📦 [Detector] Using ONNX weights: {config.onnx_weights}")
            return config.onnx_weights
        
        # Check standard fallback locations
        fallbacks = [
            "runs/detect/train-7/weights/best.pt",
            "runs/detect/train-8/weights/best.pt",
            "runs/detect/train/weights/best.pt",
            "best.onnx",
            "yolo26n.pt",
            "yolov8n.pt",
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                print(f"⚠️ [Detector] Specified model not found. Using fallback: {fb}")
                return fb

        raise FileNotFoundError(f"❌ [Detector] Could not find any valid YOLO weights at {config.weights}")

    def _load_model(self):
        """Load YOLO model."""
        print(f"🧠 [Detector] Loading YOLO model: {self.model_path} (Device: {self.device})")
        self.model = YOLO(self.model_path)
        print(f"✅ [Detector] Model loaded. Classes: {self.model.names}")

    def predict(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Run inference on frame and return list of detected objects.
        Filters specifically for target_class ('TEL').
        """
        if self.model is None or frame is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.config.conf_threshold,
            device=self.device,
            verbose=False
        )

        detected_objects: List[DetectedObject] = []
        if not results or len(results) == 0:
            return detected_objects

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detected_objects

        target_name_lower = self.config.target_class.strip().lower()

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            cls_name = self.model.names.get(cls_id, str(cls_id))

            # Filter for TEL class (match by name or if only 1 class in model)
            is_target = (cls_name.lower() == target_name_lower) or (len(self.model.names) == 1) or ("tel" in cls_name.lower())
            if not is_target:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w = x2 - x1
            h = y2 - y1

            detected_objects.append(DetectedObject(
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                box=(x1, y1, x2, y2),
                conf=conf_score,
                cls_id=cls_id,
                cls_name=cls_name
            ))

        return detected_objects
