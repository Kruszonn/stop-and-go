import cv2
import threading

class Camera:

    def __init__(self, src=0, width=320, height=240, fps=30):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self.frame   = None
        self.lock    = threading.Lock()
        self.running = False
        self._thread: threading.Thread | None = None  # <-- adnotacja typu
    def _reader(self):              # <-- przeniesione PRZED start()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.frame = frame

    def start(self):                # <-- teraz start() widzi już _reader
        self.running = True
        self._thread = threading.Thread(
            target=self._reader, daemon=True
        )
        self._thread.start()
        return self

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def get_frame_color(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join()
        self.cap.release()

    def is_ok(self):
        return self.cap.isOpened()