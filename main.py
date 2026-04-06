import time
import signal
import sys

from camera import Camera
from detector import Detector
from distance import DistanceEstimator
from STATE_MACHINE import StopANDGoFSM, State
from driver import MotorDriver
from controller import DriverControll

FRAME_W = 320
FRAME_H = 240
W_REAL = 0.12
CALIBFILE = 'camera_calibration.npz'
SIMULATE = True

LOOP_HZ = 30
LOOP_DT = 1.0/LOOP_HZ

cam = Camera(width=FRAME_W, height=FRAME_H).start()

det = Detector(frame_width=FRAME_W, frame_height=FRAME_H, tracker_type='KCF')

dist = DistanceEstimator(calibration_file=CALIBFILE, vehicle_real_width_m=W_REAL)
dist.init_undistort_maps(FRAME_W, FRAME_H)

fsm = StopANDGoFSM()
ctrl = DriverControll(d_slow=StopANDGoFSM.D_SLOW, d_stop=StopANDGoFSM.D_STOP)

drv = MotorDriver(simulate=SIMULATE)

d_history : list[float] = []


def shutdown(sig=None, frame=None):
    print("\n[SHUTDOWN] zatrzymwanie pojazdu...")
    drv.stop()
    drv.cleanup()
    cam.stop()
    sys.exit()
signal.signal(signal.SIGINT, shutdown)

print("[START] System Stop & Go uruchomiony")
print(f"  Tryb: {'SYMULACJA' if SIMULATE else 'SPRZĘT'}")
print(f"  Rozdzielczość: {FRAME_W}×{FRAME_H}")
print(f"  W_real: {W_REAL} m")
print(f"  Pętla: {LOOP_HZ} Hz\n")


while True:
    t_start = time.time()
    gray = cam.get_frame()
    if gray is None:
        time.sleep(0.01)
        continue
    
    # 2. Korekcja zniekształceń
    gray = dist.undistort(gray)
    # 3. Detekcja pojazdu
    bbox = det.update(gray)
    # 4. Estymacja odległości z wygładzaniem
    d = dist.estimate_with_smoothing(bbox, d_history, window=5)
    # 5. Maszyna stanów
    state = fsm.update(d)
    # 6. Kontroler prędkości → PWM
    pwm = ctrl.update(state, d)
    # 7. Wyślij do silnika
    drv.set_speed(pwm)
    # 8. Log tylko przy zmianie stanu
    if fsm.changed:
        d_str = f"{d:.3f} m" if d is not None else "brak"
        print(f"[STAN] {state.name:<12} | d={d_str} | PWM={pwm:.2f}")

    # 9. Regulacja częstotliwości pętli
    elapsed = time.time() - t_start
    sleep_t = LOOP_DT - elapsed
    if sleep_t > 0:
        time.sleep(sleep_t)