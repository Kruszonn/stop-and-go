from enum import Enum, auto
import time


class State(Enum):
    DRIVING = auto()
    SLOWING = auto()
    STOP = auto()
    MOVING = auto()

class StopANDGoFSM:
    D_SAFE = 0.40 #m
    D_SLOW = 0.25 #m
    D_STOP = 0.12 #m
    D_HYST = 0.18 #m

    WAIT_FOR_DRIVE = 0.5

    def __init__(self):
        self._state = State.DRIVING
        self._prev_state = None
        self._stop_since = None

    def update(self, distance_m: float | None) -> State:
        self._prev_state = self._state

        if distance_m is None:
            if self._state == State.DRIVING:
                self._state = State.STOP
            return self._state

        if self._state == State.DRIVING:
            self._from_driving(distance_m)
        elif self._state == State.SLOWING:
            self._from_slowing(distance_m)
        elif self._state == State.STOP:
            self._from_stop(distance_m)
        elif self._state == State.MOVING:
            self._from_moving(distance_m)
        return self._state
    
    def _from_driving(self, d: float):
        if d <= self.D_STOP:
            self._enter_stop()
        elif d <= self.D_SLOW:
            self._state = State.SLOWING

    def _from_slowing(self, d: float):
        if d <= self.D_STOP:
            self._enter_stop()
        elif d > self.D_SAFE:
            self._state = State.DRIVING

    def _from_stop(self, d: float):
        if self._stop_since is None:
            return
        elapsed = time.time() - self._stop_since
        if elapsed < self.WAIT_FOR_DRIVE:
            return
        
        if d > self.D_HYST:
            self._state = State.MOVING

    def _from_moving(self, d: float):
        if d <= self.D_STOP:
            self._enter_stop()
        elif d <= self.D_SLOW:
            self._state = State.SLOWING
        elif d > self.D_SAFE:
            self._state = State.DRIVING

    def _enter_stop(self):
        self._state = State.STOP
        self._stop_since = time.time()

    @property
    def state(self) -> State:
        return self._state

    @property
    def changed(self) -> bool:
        return self._state != self._prev_state
    




