import time
import json
import threading
import queue
import requests
from typing import Optional, Dict, Any
from .config import HttpSenderConfig

class HttpSender:
    """
    Asynchronous, non-blocking HTTP POST sender.
    Uses a background daemon thread and a single-slot queue to send the latest
    vision detection telemetry to TELBot-Host without blocking the camera frame loop.
    """
    def __init__(self, config: HttpSenderConfig):
        self.config = config
        self.enabled = config.enabled
        self.endpoint_url = config.endpoint_url
        self.timeout = config.timeout
        self.min_interval = 1.0 / max(1.0, config.max_fps)

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_send_time = 0.0

        if self.enabled and self.endpoint_url:
            self._start_worker()

    def _start_worker(self):
        """Start the background HTTP posting worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="HttpSenderWorker")
        self._thread.start()
        print(f"📡 [HttpSender] Background sender started. Target: {self.endpoint_url}")

    def send_telemetry(
        self,
        active_target: str,
        target_detected: bool,
        fire_control_state: str,
        target_coords: Optional[tuple] = None,
        column_index: int = -1
    ):
        """
        Queue telemetry for sending. Drops older pending items to ensure only the
        latest real-time state is transmitted.
        """
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_send_time < self.min_interval:
            return

        payload: Dict[str, Any] = {
            "timestamp": now,
            "active_target": active_target,          # "left", "center", "right", "none"
            "target_detected": target_detected,      # True if 3 TEL markers acquired
            "fire_control_state": fire_control_state # "READY" or "HOLD"
        }

        # Optional supplemental fields for richer integration with TELBot-Host
        if target_coords is not None:
            payload["target_x"] = int(target_coords[0])
            payload["target_y"] = int(target_coords[1])
        if column_index >= 0:
            payload["column_index"] = column_index

        # Keep only the newest payload in queue
        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(payload)
            self._last_send_time = now
        except queue.Full:
            pass

    def _worker_loop(self):
        """Worker thread loop to perform HTTP POST requests."""
        while self._running:
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                if response.status_code >= 400:
                    # Endpoint returned an error
                    pass
            except requests.RequestException:
                # Host is offline or endpoint not yet configured; silently handle to prevent logging spam
                pass
            except Exception as e:
                print(f"⚠️ [HttpSender] Unexpected error during POST: {e}")
            finally:
                self._queue.task_done()

    def stop(self):
        """Gracefully stop the worker thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        print("🔌 [HttpSender] Background sender stopped.")
