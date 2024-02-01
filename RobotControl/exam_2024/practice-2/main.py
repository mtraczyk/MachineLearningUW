import mujoco
from mujoco import viewer
from car_simulator import CarSimulator
from pid import PID


def simulate(cs, desired_pos):
    # TODO: design control for the system
    pid_gps = PID(
        gain_prop=7., gain_int=0., gain_der=0.2, sensor_period=100
    )

    pid_ang_vel = PID(
        gain_prop=7., gain_int=0.2, gain_der=0., sensor_period=1
    )
    # TODO end

    for i in range(3_000):
        # TODO: find value for `forward_torque` using designed control
        if i % 100 == 0:
            desired_ang_vel = pid_gps.output_signal(desired_pos, cs.position)
        current_ang_vel = cs.wheel_angular_vel
        forward_torque = pid_ang_vel.output_signal(desired_ang_vel, current_ang_vel)
        # TODO end

        print('-' * 20)
        print(f'Desired ang velocity: {desired_ang_vel}')
        print(f'Gyro: {cs.wheel_angular_vel}')
        print(f'GPS: {cs.position}')
        print(f'Forward_torque: {forward_torque}')
        print()

        cs.sim_step(forward_torque)

    cs.viewer.close()


if __name__ == '__main__':
    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    cs = CarSimulator(gps_freq=0.01, rendering_freq=1)
    desired_pos = cs.data.body("traffic_light_gate").xpos[1]

    simulate(cs=cs, desired_pos=desired_pos)
