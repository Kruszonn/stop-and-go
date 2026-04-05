import cv2
import numpy as np


class DistanceEstimator:
    """
    Estymuje odległość do pojazdu poprzedzającego
    na podstawie szerokości bbox w pikselach.

    Wzór: d = (W_real * focal_length_px) / W_pixel
    """

    def __init__(self,
                 calibration_file: str,
                 vehicle_real_width_m: float = 1.8):
        self.W_real = vehicle_real_width_m

        data = np.load(calibration_file)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs   = data['dist_coeffs']

        self.focal_length_px = float(self.camera_matrix[0, 0])

        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None

    def init_undistort_maps(self, frame_width: int,
                             frame_height: int):
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (frame_width, frame_height),
            cv2.CV_16SC2
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self._map1 is None:
            h, w = frame.shape[:2]
            self.init_undistort_maps(w, h)
        return cv2.remap(frame, self._map1, self._map2,
                         cv2.INTER_LINEAR)

    def estimate(self, bbox) -> float | None:
        if bbox is None:
            return None

        _, _, W_pixel, _ = bbox

        if W_pixel < 5:
            return None

        distance = (self.W_real * self.focal_length_px) / W_pixel
        return round(distance, 3)

    def estimate_with_smoothing(self, bbox,
                                 history: list,
                                 window: int = 5) -> float | None:
        d = self.estimate(bbox)

        if d is not None:
            history.append(d)
            if len(history) > window:
                history.pop(0)

        if not history:
            return None

        return float(np.median(history))

    @property
    def focal_length(self) -> float:
        return self.focal_length_px