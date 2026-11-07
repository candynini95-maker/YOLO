import os
import yaml
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional

@dataclass
class VideoConfig:
    source: Union[str, int] = "http://127.0.0.1:8080/stream.mjpg"
    enable_fallback: bool = True
    fallback_device: int = 0
    reconnect_delay_sec: float = 2.0
    frame_width: int = 0
    frame_height: int = 0

@dataclass
class ModelConfig:
    weights: str = "runs/detect/train-7/weights/best.pt"
    onnx_weights: str = "best.onnx"
    device: str = "auto"
    conf_threshold: float = 0.5
    target_class: str = "TEL"

@dataclass
class GeometryConfig:
    row_ratios: List[float] = field(default_factory=lambda: [2.0, 6.0, 10.0, 14.0])
    roi_sample_radius: int = 14
    dynamic_mask_radius: int = 20

@dataclass
class ColorFilterConfig:
    lower_red1: List[int] = field(default_factory=lambda: [0, 30, 150])
    upper_red1: List[int] = field(default_factory=lambda: [10, 255, 255])
    lower_red2: List[int] = field(default_factory=lambda: [170, 30, 150])
    upper_red2: List[int] = field(default_factory=lambda: [180, 255, 255])
    min_red_pixel_count: int = 50

@dataclass
class HttpSenderConfig:
    enabled: bool = True
    endpoint_url: str = "http://127.0.0.1:8000/api/vision/target"
    timeout: float = 0.3
    max_fps: float = 20.0

@dataclass
class SystemConfig:
    headless: bool = False
    hud_window_name: str = "TEL Robot - Vision HUD"
    debug_window_name: str = "TEL Robot - HSV Mask Debug"

@dataclass
class VisionConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    color_filter: ColorFilterConfig = field(default_factory=ColorFilterConfig)
    http_sender: HttpSenderConfig = field(default_factory=HttpSenderConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "VisionConfig":
        """Load configuration from a YAML file, merging with default values."""
        if not os.path.exists(path):
            print(f"⚠️ [Config] Config file not found at {path}, using defaults.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        video_data = data.get("video", {})
        model_data = data.get("model", {})
        geometry_data = data.get("geometry", {})
        color_data = data.get("color_filter", {})
        http_data = data.get("http_sender", {})
        system_data = data.get("system", {})

        return cls(
            video=VideoConfig(**{k: v for k, v in video_data.items() if hasattr(VideoConfig, k)}),
            model=ModelConfig(**{k: v for k, v in model_data.items() if hasattr(ModelConfig, k)}),
            geometry=GeometryConfig(**{k: v for k, v in geometry_data.items() if hasattr(GeometryConfig, k)}),
            color_filter=ColorFilterConfig(**{k: v for k, v in color_data.items() if hasattr(ColorFilterConfig, k)}),
            http_sender=HttpSenderConfig(**{k: v for k, v in http_data.items() if hasattr(HttpSenderConfig, k)}),
            system=SystemConfig(**{k: v for k, v in system_data.items() if hasattr(SystemConfig, k)}),
        )

