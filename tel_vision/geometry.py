import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from .detector import DetectedObject
from .config import GeometryConfig

@dataclass
class RigidBodyTarget:
    acquired: bool = False
    tels: List[DetectedObject] = field(default_factory=list)  # Sorted: 0=Left, 1=Center, 2=Right
    row1_holes: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # col_idx -> (x, y)
    all_holes: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)  # (row_idx, col_idx) -> (x, y)
    board_width: float = 0.0
    unit_vector: Tuple[float, float] = (1.0, 0.0)
    normal_vector: Tuple[float, float] = (0.0, 1.0)
    dynamic_mask: Optional[np.ndarray] = None

class GeometryProjector:
    """
    Module for rigid-body geometric vector calculations and hole projection.
    Uses 3 detected TEL markers to compute the 2D plane orientation and projects
    the 12 hole positions on the target board.
    """
    def __init__(self, config: GeometryConfig):
        self.config = config

    def project(self, tels: List[DetectedObject], frame_shape: Tuple[int, int, int]) -> RigidBodyTarget:
        """
        Calculate rigid body vector orientation and project 12 holes.
        Requires exactly 3 TEL markers (or select top 3 confident if >= 3).
        """
        h_img, w_img = frame_shape[0], frame_shape[1]

        if len(tels) < 3:
            return RigidBodyTarget(acquired=False)

        # Select the 3 TEL markers with highest confidence if more than 3 detected
        if len(tels) > 3:
            tels = sorted(tels, key=lambda t: t.conf, reverse=True)[:3]

        # Sort left to right by cx
        sorted_tels = sorted(tels, key=lambda t: t.cx)

        # Calculate board baseline vector between left and right TEL
        dx = sorted_tels[2].cx - sorted_tels[0].cx
        dy = sorted_tels[2].cy - sorted_tels[0].cy
        board_w = float(np.sqrt(dx**2 + dy**2))

        if board_w < 10:
            return RigidBodyTarget(acquired=False)

        # Normalized base vector u (along the board)
        ux = dx / board_w
        uy = dy / board_w

        # Normal vector pointing downwards (perpendicular to u)
        down_x = -uy
        down_y = ux

        row1_holes: Dict[int, Tuple[int, int]] = {}
        all_holes: Dict[Tuple[int, int], Tuple[int, int]] = {}
        dynamic_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # Project 3 columns x 4 rows = 12 holes
        for c_idx, t in enumerate(sorted_tels):
            for r_idx, ratio in enumerate(self.config.row_ratios):
                hole_x = int(t.cx + (ratio * t.h * down_x))
                hole_y = int(t.cy + (ratio * t.h * down_y))

                all_holes[(r_idx, c_idx)] = (hole_x, hole_y)
                if r_idx == 0:
                    row1_holes[c_idx] = (hole_x, hole_y)

                # Draw circle on dynamic mask
                cv2.circle(
                    dynamic_mask,
                    (hole_x, hole_y),
                    self.config.dynamic_mask_radius,
                    255,
                    -1
                )

        return RigidBodyTarget(
            acquired=True,
            tels=sorted_tels,
            row1_holes=row1_holes,
            all_holes=all_holes,
            board_width=board_w,
            unit_vector=(ux, uy),
            normal_vector=(down_x, down_y),
            dynamic_mask=dynamic_mask
        )

