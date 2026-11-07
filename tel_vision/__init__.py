"""
TEL Vision System Package
Modular vision computing pipeline for TEL Robot.
"""

from .config import VisionConfig
from .detector import YOLODetector
from .geometry import GeometryProjector, RigidBodyTarget
from .color_filter import AntiOverexposureColorFilter
from .video_stream import VideoStream
from .http_client import HttpSender
from .renderer import HUDRenderer
from .pipeline import VisionPipeline

__all__ = [
    "VisionConfig",
    "YOLODetector",
    "GeometryProjector",
    "RigidBodyTarget",
    "AntiOverexposureColorFilter",
    "VideoStream",
    "HttpSender",
    "HUDRenderer",
    "VisionPipeline",
]
