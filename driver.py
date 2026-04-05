class MotorDriver:
    def __init__(self, simulate: bool = True,
                 pin_pwm:   int = 12,
                 IN1: int = 23,
                 IN2: int = 24):
        self.simulate = simulate

        if not simulate:
            import RPi.GPIO as GPIO # type: ignore[import]
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin_pwm, GPIO.OUT)
            GPIO.setup(IN1, GPIO.OUT)
            GPIO.setup(IN2, GPIO.OUT)

            self._pwm_pin = GPIO.PWM(pin_pwm, 1000)
            self._pwm_pin.start(0)
            self._pin_a = IN1
            self._pin_b = IN2


    def set_speed(self, pwm_value: float):
        pwm_value = max(-1.0, min(1.0, pwm_value))

        if self.simulate:
            bar = '█' * int(abs(pwm_value) * 20)
            print(f"  MOTOR: {pwm_value:+.2f}  |{bar:<20}|")
            return
        
        forward = pwm_value >= 0
        self._GPIO.output(self._pin_a,
                          self._GPIO.HIGH if forward else self._GPIO.LOW)
        self._GPIO.output(self._pin_b,
                          self._GPIO.LOW  if forward else self._GPIO.HIGH)

        self._pwm_pin.ChangeDutyCycle(abs(pwm_value) * 100)


    def stop(self):
        self.set_speed(0.0)


    def cleanup(self):
        if not self.simulate:
            self._pwm_pin.stop()
            self._GPIO.cleanup()

