class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: initialize additional attributes
        self.previous_sensor_reading = None
        self.integral = 0

    def output_signal(self, commanded_variable, sensor_reading):
        # TODO: compute output signal of the controller
        error = commanded_variable - sensor_reading
        self.integral += error * self.sensor_period
        derivative = 0
        if self.previous_sensor_reading is not None:
            derivative = (sensor_reading - self.previous_sensor_reading) * self.sensor_period
        self.previous_sensor_reading = sensor_reading

        return self.gain_prop * error - self.gain_der * derivative + self.gain_int * self.integral
