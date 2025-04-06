"""
Utility functions for the project.
"""

import time
import cv2


_prev_time = None

def update_fps(frame):
    global _prev_time
    curr_time = time.time()
    if _prev_time is not None:
        elapsed = curr_time - _prev_time
        if elapsed > 0:
            fps = 1 / elapsed
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    _prev_time = curr_time