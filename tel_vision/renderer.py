import cv2
import numpy as np
from typing import Optional, Tuple
from .geometry import RigidBodyTarget
from .color_filter import ColorFilterResult
from .config import SystemConfig

class HUDRenderer:
    """
    Tactical HUD and visual debugger renderer.
    Completely decoupled from the computer vision computation core.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.hud_window = config.hud_window_name
        self.debug_window = config.debug_window_name

    def render(
        self,
        frame: np.ndarray,
        target: RigidBodyTarget,
        color_result: ColorFilterResult,
        fire_control_state: str,
        fps: float = 0.0
    ) -> np.ndarray:
        """Render the tactical HUD overlay on top of the raw BGR frame."""
        display = frame.copy()
        h_img, w_img = frame.shape[0], frame.shape[1]
        cx_img, cy_img = w_img // 2, h_img // 2

        # 1. Central tactical crosshair
        cv2.drawMarker(display, (cx_img, cy_img), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(display, (cx_img, cy_img), 20, (0, 255, 0), 1)

        # 2. Draw detected TEL boxes if target acquired
        if target.acquired:
            for t in target.tels:
                x1, y1, x2, y2 = t.box
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 128, 0), 2)
                cv2.circle(display, (t.cx, t.cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"TEL {t.conf:.2f}",
                    (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 128, 0),
                    2
                )

            # 3. Draw 12 projected holes and target lock-on box
            for (r_idx, c_idx), (hx, hy) in target.all_holes.items():
                is_active = (r_idx == 0 and c_idx == color_result.active_column)
                if is_active:
                    # Sniper lock-on box
                    cv2.rectangle(display, (hx - 18, hy - 18), (hx + 18, hy + 18), (0, 0, 255), 3)
                    cv2.drawMarker(display, (hx, hy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=3)
                    label = f"R1-{color_result.active_column_name.upper()} [ SCORE! ]"
                    cv2.putText(display, label, (hx - 45, hy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                else:
                    cv2.drawMarker(display, (hx, hy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
                    cv2.putText(
                        display,
                        f"R{r_idx+1}-C{c_idx+1}",
                        (hx - 15, hy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1
                    )

        # 4. Top-left HUD telemetry panel
        cv2.rectangle(display, (15, 15), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(display, (15, 15), (300, 100), (0, 255, 0), 1)
        cv2.putText(display, "SYS: TEL ROBOT VISION v3.0", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fps_text = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        cv2.putText(display, f"STATUS: STREAMING | {fps_text}", (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        col_str = color_result.active_column_name.upper() if color_result.active_column != -1 else "NONE"
        cv2.putText(display, f"ACTIVE COL : {col_str}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 5. Bottom target acquisition state
        if not target.acquired:
            cv2.putText(display, "SCANNING TARGET...", (20, h_img - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display, "TARGET ACQUIRED", (20, h_img - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 6. Fire control status banner
        if fire_control_state == 'READY':
            cv2.rectangle(display, (w_img - 220, h_img - 60), (w_img - 20, h_img - 20), (0, 255, 0), -1)
            cv2.putText(display, "READY TO FIRE", (w_img - 200, h_img - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.rectangle(display, (w_img - 240, h_img - 60), (w_img - 20, h_img - 20), (0, 0, 255), -1)
            cv2.putText(display, "FIRE CONTROL HOLD", (w_img - 230, h_img - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return display

    def show_frames(self, hud_frame: np.ndarray, debug_mask: Optional[np.ndarray]) -> bool:
        """
        Display GUI windows and check for exit key ('q').
        Returns False if user pressed 'q' to quit, True otherwise.
        """
        cv2.imshow(self.hud_window, hud_frame)
        if debug_mask is not None:
            cv2.imshow(self.debug_window, debug_mask)

        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')

    def close_all(self):
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
