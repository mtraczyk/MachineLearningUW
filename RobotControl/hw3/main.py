import numpy as np
import cv2
import datetime
import argparse

process_var = 1  # Process noise variance
measurement_var = 1e4  # Measurement noise variance


class KalmanFilter:
    def __init__(self, process_var, measurement_var_x, measurement_var_y):
        # process_var: process variance, represents uncertainty in the model
        # measurement_var: measurement variance, represents measurement noise

        ### TODO
        ### Change the model to constant acceleration model

        # Measurement Matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Process Covariance Matrix
        self.Q = np.eye(4) * process_var

        # Measurement Covariance Matrix
        self.R = np.array(
            [
                [measurement_var_x, 0],
                [0, measurement_var_y],
            ]
        )

        # Initial State Covariance Matrix
        self.P = np.eye(4)

        # Initial State
        self.x = np.zeros(4)

    def predict(self, dt):
        # State Transition Matrix
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Predict the next state
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q
        print(f"Predicted State: {self.x}")

    def update(self, measurement):
        # Update the state with the new measurement
        print(f"Measurement: {measurement}")
        y = measurement - self.H @ self.x
        print(f"y: {y}")
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


def draw_uncertainty(kf, img):
    ### TODO
    ### Draw uncertainty
    pass


class ClickReader:
    def __init__(self, process_var, measurement_var, window_name="Click Window"):
        self.window_name = window_name
        self.cur_time = datetime.datetime.now()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.kf = KalmanFilter(process_var, measurement_var, measurement_var)

        self.img = 255 * np.ones((500, 500, 3), np.uint8)

    def mouse_callback(self, event, x, y, flags, param):
        # Check if the event is a left button click
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Time: {current_time}, Position: ({x}, {y})")
            new_time = datetime.datetime.now()
            self.kf.predict((new_time - self.cur_time).total_seconds())
            self.cur_time = new_time

            cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)  # Red color, filled circle
            self.kf.update(np.array((x, y)))
            print(f"Updated State: {self.kf.x}")

    def run(self):
        # Main loop to display the window
        while True:
            new_time = datetime.datetime.now()
            self.kf.predict((new_time - self.cur_time).total_seconds())
            self.cur_time = new_time

            cv2.circle(
                self.img, (int(self.kf.x[0]), int(self.kf.x[1])), 2, (255, 0, 0), -1
            )  # Blue color, filled circle

            img_draw = self.img.copy()

            draw_uncertainty(self.kf, img_draw)

            cv2.imshow(self.window_name, img_draw)

            # Exit on pressing the 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


class PredefinedClickReader:
    def __init__(
        self,
        process_var,
        measurement_var_x,
        measurement_var_y,
        window_name="Click Window",
    ):
        self.window_name = window_name
        self.cur_time = datetime.datetime.now()
        cv2.namedWindow(self.window_name)
        self.kf = KalmanFilter(process_var, measurement_var_x, measurement_var_y)

        self.img = 255 * np.ones((500, 500, 3), np.uint8)

    def run(self, observation_generator):
        for dt, observation in observation_generator:
            self.kf.predict(dt)
            if observation is not None:
                self.kf.update(observation)
                cv2.circle(
                    self.img,
                    (int(observation[0]), int(observation[1])),
                    2,
                    (0, 0, 255),
                    -1,
                )
            cv2.circle(
                self.img, (int(self.kf.x[0]), int(self.kf.x[1])), 2, (255, 0, 0), -1
            )
            img_draw = self.img.copy()
            draw_uncertainty(self.kf, img_draw)
            cv2.imshow(self.window_name, img_draw)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parabola_generator():
    for x in range(0, 500, 1):
        if np.random.rand(1)[0] > 0.5:
            yield 1, None
        else:
            yield 1, np.array(
                [
                    x + np.random.randn(1)[0] * np.sqrt(1e2),
                    x * (500 - x) / 250 + np.random.randn(1)[0] * np.sqrt(4e2),
                ]
            )


class VideoReader:
    def __init__(
        self,
        process_var,
        measurement_var,
        video_path,
        window_name="Video Window",
        fps=29.97,
    ):
        self.video = cv2.VideoCapture(video_path)
        self.kf = KalmanFilter(process_var, measurement_var, measurement_var)
        self.window_name = window_name
        cv2.namedWindow(window_name)
        self.fps = fps

    def run(self):
        ### TODO
        ### Set initial position using the first frame
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            ### TODO
            ### Find the red ball in the frame
            ### Use Kalman Filter to track the ball and predict its position
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # add an argument to decide between click, predefined and video
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="click",
        choices=["click", "predefined", "video"],
        help="Mode to run the program in. Options are click, predefined, video",
    )
    args = parser.parse_args()
    if args.mode == "click":
        click_reader = ClickReader(process_var, measurement_var)
        click_reader.run()
    elif args.mode == "predefined":
        ### TODO
        ### Read parabola_generator and set measurement_var_x and measurement_var_y

        predefinedclicker = PredefinedClickReader(0, ...)
        predefinedclicker.run(parabola_generator())
    else:
        assert args.mode == "video"
        video_reader = VideoReader(10, 10, "line.mp4")
        video_reader.run()
