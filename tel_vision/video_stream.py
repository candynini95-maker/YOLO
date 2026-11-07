import cv2
import time
from typing import Union, Tuple, Optional
import numpy as np
from .config import VideoConfig

class VideoStream:
    """
    Video stream handler supporting HTTP MJPEG streams, RTSP streams, and local webcams.
    Includes automatic reconnection logic.
    """
    def __init__(self, config: VideoConfig):
        self.config = config
        self.source = config.source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.consecutive_failures = 0
        self.max_failures = 30
        self.using_fallback = False

        self._init_capture()

    def _init_capture(self) -> bool:
        """Initialize the VideoCapture object."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        target = self.source
        if isinstance(target, str) and target.isdigit():
            target = int(target)

        # Detect source type
        is_device_index = isinstance(target, int)
        is_stream_url = isinstance(target, str) and (target.startswith("http://") or target.startswith("https://") or target.startswith("rtsp://"))

        print(f"📡 [VideoStream] Connecting to source: {target} (Stream URL: {is_stream_url}, Index: {is_device_index})")

        try:
            if is_device_index:
                # Local camera on Windows prefers CAP_DSHOW
                self.cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            else:
                # HTTP MJPEG or RTSP network stream
                self.cap = cv2.VideoCapture(target)

            if self.cap.isOpened():
                if self.config.frame_width > 0 and self.config.frame_height > 0:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self.is_opened = True
                self.consecutive_failures = 0
                print(f"✅ [VideoStream] Successfully opened: {target}")
                return True
            else:
                print(f"⚠️ [VideoStream] Failed to open: {target}")
        except Exception as e:
            print(f"❌ [VideoStream] Error initializing video source {target}: {e}")

        # Try fallback if enabled
        if self.config.enable_fallback and not self.using_fallback:
            fallback = self.config.fallback_device
            print(f"🔄 [VideoStream] Attempting fallback to local camera: {fallback}")
            self.using_fallback = True
            try:
                self.cap = cv2.VideoCapture(fallback, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    self.is_opened = True
                    self.consecutive_failures = 0
                    print(f"✅ [VideoStream] Fallback camera {fallback} opened successfully.")
                    return True
            except Exception as e:
                print(f"❌ [VideoStream] Fallback camera also failed: {e}")

        self.is_opened = False
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the stream.
        Attempts auto-reconnect if frames fail consecutively.
        """
        if not self.is_opened or self.cap is None:
            time.sleep(self.config.reconnect_delay_sec)
            self._init_capture()
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                print(f"⚠️ [VideoStream] Stream interrupted ({self.consecutive_failures} failures). Reconnecting...")
                self.is_opened = False
                time.sleep(self.config.reconnect_delay_sec)
                self._init_capture()
            return False, None

        self.consecutive_failures = 0
        return True, frame

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        print("🔌 [VideoStream] Released video capture.")

