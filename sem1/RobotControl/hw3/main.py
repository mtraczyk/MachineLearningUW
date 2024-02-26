import numpy as np
import cv2
import datetime
import argparse

from scipy.stats import chi2

process_var = 1  # Process noise variance
measurement_var = 1e4  # Measurement noise variance


class KalmanFilter:
    def __init__(self, process_var, measurement_var_x, measurement_var_y):
        # process_var: process variance, represents uncertainty in the model
        # measurement_var: measurement variance, represents measurement noise

        # Measurement Matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        # Process Covariance Matrix
        self.Q = np.eye(6) * process_var

        # Measurement Covariance Matrix
        self.R = np.array(
            [
                [measurement_var_x, 0],
                [0, measurement_var_y],
            ]
        )

        # Initial State Covariance Matrix
        self.P = np.eye(6)

        # Initial State
        self.x = np.zeros(6)

    def predict(self, dt):
        # State Transition Matrix
        A = np.array([[1, 0, dt, 0, 0, 0], [0, 1, 0, dt, 0, 0],
                      [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt],
                      [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

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
        self.P = (np.eye(6) - K @ self.H) @ self.P


def draw_uncertainty(kf, img):
    state_mean = kf.x[:2]
    state_cov = kf.P[:2, :2]
    # Draw the uncertainty ellipse, 90% confidence
    eigenvalues, eigenvectors = np.linalg.eig(state_cov)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle = 180.0 * angle / np.pi
    # https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    chi_square_quantile = chi2.ppf(0.9, 2)
    major = np.sqrt(chi_square_quantile) * np.sqrt(eigenvalues[0])
    minor = np.sqrt(chi_square_quantile) * np.sqrt(eigenvalues[1])
    cv2.ellipse(
        img,
        (int(state_mean[0]), int(state_mean[1])),
        (int(major), int(minor)),
        int(angle),
        0,
        360,
        (0, 0, 255),
        2,
    )


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


def find_red_ball(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red color
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # finding contour with maximum area and store it as best_cnt
    max_area = 0.
    best_cnt = None
    if contours is None:
        return None

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            best_cnt = c

    # finding centroids of best_cnt and draw a circle there
    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    return cx, cy


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
        _, frame = self.video.read()
        cx, cy = find_red_ball(frame)
        self.kf.x = np.array([cx, cy, 0, 0, 0, 0])

        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            ### Find the red ball in the frame
            ### Use Kalman Filter to track the ball and predict its position

            self.kf.predict(1 / self.fps)
            measurement = find_red_ball(frame)
            if measurement is not None:
                self.kf.update(np.array(measurement))

            cv2.circle(
                frame, (int(self.kf.x[0]), int(self.kf.x[1])), 15, (255, 0, 0), -1
            )
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
        predefinedclicker = PredefinedClickReader(0.,  100., 100.)
        predefinedclicker.run(parabola_generator())
    else:
        assert args.mode == "video"
        video_reader = VideoReader(1000, 100, "sinewave.mp4")
        video_reader.run()
