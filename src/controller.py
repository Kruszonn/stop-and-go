import time 
from STATE_MACHINE import State

class DriverControll:
    V_SET = 0.55
    V_MIN = 0.20
    V_START = 0.25
    RAMP_UP = 0.40
    RAMP_DOWN = 0.80

    def __init__(self, d_slow: float, d_stop: float):
        self.d_slow = d_slow
        self.d_stop = d_stop

        self._pwm = 0.0
        self._last_time = time.time()

    def update(self, state: State, distance_m: float | None) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        target = self._target_pwm(state, distance_m)
        self._pwm = self._ramp(self._pwm, target, dt)
        return round(self._pwm, 3)
    
    def _target_pwm(self, state: State, distance_m: float | None) -> float:
        if state == State.DRIVING:
            return self.V_SET
        elif state == State.STOP:
            return 0.0
        elif state == State.MOVING:
            return self.V_START
        elif state == State.SLOWING:
            return self._proportional_speed(distance_m)
        return 0.0
    
    def _proportional_speed(self, d: float | None) -> float:
        if d is None:
            return 0.0
        
        ratio = (d - self.d_stop) / (self.d_slow - self.d_stop)
        ratio = max(0.0, min(1.0, ratio))
        return self.V_MIN + ratio * (self.V_SET - self.V_MIN)
    
    def _ramp(self, current: float, target: float, dt: float) -> float:
        if target > current:
            max_step = self.RAMP_UP * dt
        else:
            max_step = self.RAMP_DOWN * dt

        diff = target - current
        if abs(diff) <= max_step:
            return target
        return current + max_step * (1 if diff > 0 else -1)

    @property
    def pwm(self) -> float:
        return self._pwm
    
    