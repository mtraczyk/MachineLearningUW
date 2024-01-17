class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        self.previous_error = None
        self.integral = 0

    def output_signal(self, commanded_variable, sensor_readings):
        error = sensor_readings[-1] - commanded_variable
        self.integral += error * self.sensor_period
        derivative = 0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / self.sensor_period
        self.previous_error = error

        return -(self.gain_prop * error + self.gain_der * derivative + self.gain_int * self.integral)
