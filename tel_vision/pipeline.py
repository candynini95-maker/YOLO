import time
import cv2
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .config import VisionConfig
from .video_stream import VideoStream
from .detector import YOLODetector, DetectedObject
from .geometry import GeometryProjector, RigidBodyTarget
from .color_filter import AntiOverexposureColorFilter, ColorFilterResult
from .http_client import HttpSender
from .renderer import HUDRenderer

@dataclass
class VisionCycleResult:
    target_acquired: bool
    active_target: str       # "left", "center", "right", "none"
    fire_control_state: str  # "READY", "HOLD"
    target_coords: Optional[tuple]
    active_column: int       # 0: Left, 1: Center, 2: Right, -1: None
    fps: float
    hud_frame: Optional[np.ndarray] = None
    debug_mask: Optional[np.ndarray] = None

class VisionPipeline:
    """
    Main vision computing pipeline orchestrator.
    Coordinates video streaming, YOLO detection, rigid-body geometry projection,
    anti-overexposure color filtering, HTTP telemetry dispatch, and optional HUD rendering.
    """
    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.stream: Optional[VideoStream] = None
        self.detector = YOLODetector(self.config.model)
        self.geometry = GeometryProjector(self.config.geometry)
        self.color_filter = AntiOverexposureColorFilter(
            self.config.color_filter,
            roi_radius=self.config.geometry.roi_sample_radius
        )
        self.http_sender = HttpSender(self.config.http_sender)
        self.renderer: Optional[HUDRenderer] = None

        if not self.config.system.headless:
            self.renderer = HUDRenderer(self.config.system)

    def process_frame(self, frame: np.ndarray) -> VisionCycleResult:
        """
        Execute a single vision processing cycle on a provided BGR frame.
        Can be called directly when embedding into external host systems (e.g. TELBot-Host).
        """
        h_img, w_img = frame.shape[:2]

        # 1. Detect TEL markers
        detected_tels = self.detector.predict(frame)

        # 2. Geometric rigid-body projection
        target = self.geometry.project(detected_tels, frame.shape)

        # 3. Anti-overexposure HSV red light evaluation on Row 1
        color_result = self.color_filter.evaluate_target(frame, target)

        # 4. Fire control decision logic
        fire_control_state = "READY" if (target.acquired and color_result.active_column != -1) else "HOLD"

        target_coords = None
        if target.acquired and color_result.active_column != -1:
            target_coords = target.row1_holes.get(color_result.active_column)

        # 5. Non-blocking HTTP telemetry dispatch
        self.http_sender.send_telemetry(
            active_target=color_result.active_column_name,
            target_detected=target.acquired,
            fire_control_state=fire_control_state,
            target_coords=target_coords,
            column_index=color_result.active_column
        )

        hud_frame = None
        debug_mask = None
        if self.renderer is not None:
            hud_frame = self.renderer.render(
                frame=frame,
                target=target,
                color_result=color_result,
                fire_control_state=fire_control_state
            )
            debug_mask = color_result.debug_mask_visual

        return VisionCycleResult(
            target_acquired=target.acquired,
            active_target=color_result.active_column_name,
            fire_control_state=fire_control_state,
            target_coords=target_coords,
            active_column=color_result.active_column,
            fps=0.0,
            hud_frame=hud_frame,
            debug_mask=debug_mask
        )

    def run(self):
        """
        Run the standalone real-time vision loop from the configured video stream.
        """
        print("🎬 [VisionPipeline] Starting vision pipeline execution...")
        self.stream = VideoStream(self.config.video)

        prev_time = time.time()
        fps_smoothed = 0.0

        try:
            while True:
                ret, frame = self.stream.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Compute frame rate
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt)

                # Process single frame
                result = self.process_frame(frame)
                result.fps = fps_smoothed

                # Headless vs GUI handling
                if self.renderer is not None and result.hud_frame is not None:
                    # Update HUD with real-time FPS
                    cv2.putText(
                        result.hud_frame,
                        f"STATUS: STREAMING | FPS: {fps_smoothed:.1f}",
                        (25, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )
                    keep_running = self.renderer.show_frames(result.hud_frame, result.debug_mask)
                    if not keep_running:
                        print("🛑 [VisionPipeline] Quit signal received from GUI ('q').")
                        break

        except KeyboardInterrupt:
            print("\n🛑 [VisionPipeline] Interrupted by user (Ctrl+C).")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean up resources."""
        print("🔌 [VisionPipeline] Shutting down...")
        if self.stream is not None:
            self.stream.release()
        self.http_sender.stop()
        if self.renderer is not None:
            self.renderer.close_all()
        print("✅ [VisionPipeline] Shutdown complete.")
