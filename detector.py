import cv2
import numpy as np


class Detector:

    REDETECT_EVERY_N         = 6
    ROI_TOP_LEFT_X           = 0.25
    ROI_TOP_RIGHT_X          = 0.75
    ROI_TOP_Y                = 0.40
    ROI_BOTTOM_Y             = 0.95
    HOG_CONFIDENCE_THRESHOLD = 0.3
    MIN_BBOX_WIDTH           = 20
    MIN_BBOX_HEIGHT          = 15

    def _create_tracker(self):
        try:
            trackers = {
                'KCF':  cv2.legacy.TrackerKCF_create,   # type: ignore[attr-defined]
                'CSRT': cv2.legacy.TrackerCSRT_create,  # type: ignore[attr-defined]
            }
        except AttributeError:
            trackers = {
                'KCF':  cv2.TrackerKCF_create,   # type: ignore[attr-defined]
                'CSRT': cv2.TrackerCSRT_create,  # type: ignore[attr-defined]
            }
        if self._tracker_type not in trackers:
            raise ValueError(
                f"Nieznany tracker: {self._tracker_type}. "
                f"Dostępne: {list(trackers.keys())}"
            )
        return trackers[self._tracker_type]()

    def _build_roi_mask(self) -> np.ndarray:
        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        tl = (int(self.ROI_TOP_LEFT_X  * self.frame_w),
              int(self.ROI_TOP_Y       * self.frame_h))
        tr = (int(self.ROI_TOP_RIGHT_X * self.frame_w),
              int(self.ROI_TOP_Y       * self.frame_h))
        br = (int(0.95  * self.frame_w),
              int(self.ROI_BOTTOM_Y    * self.frame_h))
        bl = (int(0.05  * self.frame_w),
              int(self.ROI_BOTTOM_Y    * self.frame_h))
        pts = np.array([tl, tr, br, bl], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=255)  # type: ignore[call-overload]
        return mask

    def __init__(self, frame_width: int = 320, frame_height: int = 240,
                 tracker_type: str = 'KCF'):
        self.frame_w = frame_width
        self.frame_h = frame_height

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()  # type: ignore[attr-defined]
        )

        self._tracker_type = tracker_type
        self.tracker       = self._create_tracker()

        self._state       = 'SEARCHING'
        self._frame_count = 0
        self._bbox: tuple[int, int, int, int] | None = None
        self._lost_count  = 0

        self._roi_mask = self._build_roi_mask()

    def _apply_roi(self, frame_gray: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(
            frame_gray, frame_gray, mask=self._roi_mask
        )

    def _detect_hog(self, frame_gray: np.ndarray):
        roi_frame = self._apply_roi(frame_gray)
        boxes, weights = self.hog.detectMultiScale(
            roi_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        if len(boxes) == 0:
            return None

        valid: list[tuple[object, float]] = []
        for box, w in zip(boxes, weights):
            confidence = float(w) if np.ndim(w) == 0 else float(w[0])
            if (box[2] >= self.MIN_BBOX_WIDTH
                    and box[3] >= self.MIN_BBOX_HEIGHT
                    and confidence >= self.HOG_CONFIDENCE_THRESHOLD):
                valid.append((box, confidence))

        if not valid:
            return None

        best_box, _ = max(valid, key=lambda bw: bw[1])
        x, y, w, h = best_box  # type: ignore[misc]
        return (int(x), int(y), int(w), int(h))

    def _do_search(self, frame_gray: np.ndarray):
        bbox = self._detect_hog(frame_gray)
        if bbox is None:
            return None
        self.tracker = self._create_tracker()
        self.tracker.init(frame_gray, bbox)
        self._bbox        = bbox
        self._state       = 'TRACKING'
        self._frame_count = 0
        self._lost_count  = 0
        return bbox

    def _do_track(self, frame_gray: np.ndarray):
        if self._frame_count % self.REDETECT_EVERY_N == 0:
            new_bbox = self._detect_hog(frame_gray)
            if new_bbox is not None:
                self.tracker = self._create_tracker()
                self.tracker.init(frame_gray, new_bbox)
                self._bbox       = new_bbox
                self._lost_count = 0
                return new_bbox

        ok, bbox_raw = self.tracker.update(frame_gray)

        if not ok:
            self._lost_count += 1
            if self._lost_count >= 3:
                self._state = 'SEARCHING'
                self._bbox  = None
            return self._bbox

        x, y, w, h = bbox_raw
        self._bbox       = (int(x), int(y), int(w), int(h))
        self._lost_count = 0
        return self._bbox

    def update(self, frame_gray: np.ndarray):
        self._frame_count += 1
        if self._state == 'SEARCHING':
            return self._do_search(frame_gray)
        else:
            return self._do_track(frame_gray)

    def draw_debug(self, frame_color: np.ndarray) -> np.ndarray:
        out = frame_color.copy()
        contours, _ = cv2.findContours(
            self._roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (0, 200, 200), 1)
        if self._bbox is not None:
            x, y, w, h = self._bbox
            color = (0, 255, 0) if self._state == 'TRACKING' \
                    else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, f"{self._state} | w={w}px",
                        (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1)
        return out

    @property
    def state(self) -> str:
        return self._state

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._bbox