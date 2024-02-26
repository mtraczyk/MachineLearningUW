import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)
viewer.cam.distance = 4.
viewer.cam.lookat = np.array([0, 0, 1])
viewer.cam.elevation = -30.

from drone_simulator import DroneSimulator
from pid import PID

if __name__ == '__main__':
    desired_altitude = 2

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, viewer, desired_altitude=desired_altitude,
        altitude_sensor_freq=0.01, wind_change_prob=0.1, rendering_freq=1
    )

    pid_altitude = PID(
        gain_prop=0.35, gain_int=0., gain_der=0.01, sensor_period=100
    )

    pid_acceleration = PID(
        gain_prop=0.1, gain_int=100., gain_der=0., sensor_period=1
    )

    # Increase the number of iterations for a longer simulation
    for i in range(4000):
        if i % 100 == 0:
            desired_acceleration = pid_altitude.output_signal(desired_altitude, drone_simulator.measured_altitudes)
            desired_acceleration += 9.81
        current_acceleration = data.sensor("body_linacc").data[2]
        desired_thrust = pid_acceleration.output_signal(desired_acceleration, [current_acceleration])
        drone_simulator.sim_step(desired_thrust)



