import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from .config import ColorFilterConfig
from .geometry import RigidBodyTarget

COLUMN_NAMES = {
    0: "left",
    1: "center",
    2: "right",
    -1: "none"
}

@dataclass
class ColorFilterResult:
    active_column: int  # 0: Left, 1: Center, 2: Right, -1: None
    active_column_name: str  # "left", "center", "right", "none"
    red_scores: Dict[int, int]
    raw_red_mask: np.ndarray
    debug_mask_visual: Optional[np.ndarray]

class AntiOverexposureColorFilter:
    """
    Low-saturation anti-overexposure HSV color filter.
    Designed to capture the halo of high-brightness red LEDs even when the core is overexposed to white.
    Evaluates the 3 holes on Row 1 (Left, Center, Right).
    """
    def __init__(self, config: ColorFilterConfig, roi_radius: int = 14):
        self.config = config
        self.roi_radius = roi_radius

        self.lower_red1 = np.array(config.lower_red1, dtype=np.uint8)
        self.upper_red1 = np.array(config.upper_red1, dtype=np.uint8)
        self.lower_red2 = np.array(config.lower_red2, dtype=np.uint8)
        self.upper_red2 = np.array(config.upper_red2, dtype=np.uint8)

    def compute_red_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Generate binary red mask using low-saturation dual-interval HSV thresholding."""
        if img_bgr is None or img_bgr.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        return cv2.bitwise_or(mask1, mask2)

    def count_red_pixels(self, roi_bgr: np.ndarray) -> int:
        """Count red pixels in a given region of interest (ROI)."""
        mask = self.compute_red_mask(roi_bgr)
        return int(cv2.countNonZero(mask))

    def evaluate_target(self, frame: np.ndarray, target: RigidBodyTarget) -> ColorFilterResult:
        """
        Evaluate Row 1 holes of the rigid-body target to find which column (left/center/right) is lit.
        """
        h_img, w_img = frame.shape[0], frame.shape[1]
        raw_red_mask = self.compute_red_mask(frame)

        if not target.acquired:
            return ColorFilterResult(
                active_column=-1,
                active_column_name="none",
                red_scores={0: 0, 1: 0, 2: 0},
                raw_red_mask=raw_red_mask,
                debug_mask_visual=np.zeros((h_img, w_img), dtype=np.uint8)
            )

        red_scores: Dict[int, int] = {}
        for c_idx in range(3):
            if c_idx not in target.row1_holes:
                red_scores[c_idx] = 0
                continue

            hx, hy = target.row1_holes[c_idx]
            r = self.roi_radius
            y_start = max(0, hy - r)
            y_end = min(h_img, hy + r)
            x_start = max(0, hx - r)
            x_end = min(w_img, hx + r)

            roi = frame[y_start:y_end, x_start:x_end]
            red_scores[c_idx] = self.count_red_pixels(roi)

        # Find the column with maximum red score
        max_col = max(red_scores, key=red_scores.get)
        max_score = red_scores[max_col]

        # Only 1 column can be active during competition
        if max_score >= self.config.min_red_pixel_count:
            active_column = max_col
        else:
            active_column = -1

        # Generate noise-isolated debug mask using cv2.bitwise_and
        if target.dynamic_mask is not None:
            debug_mask_visual = cv2.bitwise_and(raw_red_mask, target.dynamic_mask)
        else:
            debug_mask_visual = raw_red_mask

        return ColorFilterResult(
            active_column=active_column,
            active_column_name=COLUMN_NAMES.get(active_column, "none"),
            red_scores=red_scores,
            raw_red_mask=raw_red_mask,
            debug_mask_visual=debug_mask_visual
        )
