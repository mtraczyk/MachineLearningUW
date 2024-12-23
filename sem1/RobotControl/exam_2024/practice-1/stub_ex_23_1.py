import mujoco
import time
import random
import numpy as np
import mujoco
from mujoco import viewer
import cv2

### TODO: Add your code here ###

model = mujoco.MjModel.from_xml_path("car.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def find_blue_object(img):
    # Define red color range in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue color
    mask = cv2.inRange(img, lower_blue, upper_blue)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # finding contour with maximum area and store it as best_cnt
    max_area = 0.
    best_cnt = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            best_cnt = c

    return max_area, best_cnt


def estimate_pos_of_the_blue_object(img, debug=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_area, best_cnt = find_blue_object(img_hsv)

    # finding centroids of best_cnt and draw a circle there
    M = cv2.moments(best_cnt)
    if M['m00'] == 0:
        return 0, -1, -1

    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)

    if debug:
        # save the image with the red circle around the largest red object
        cv2.imwrite("artifacts/blue_object_detection.png", img)

    return max_area, cx, cy


def get_image():
    renderer.update_scene(data, camera="camera1")
    img = renderer.render()
    return img


def check_ball(seed=1337) -> bool:
    random.seed(seed)
    steps = random.randint(0, 500)
    data.actuator("turn 1").ctrl = 1
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.actuator("turn 1").ctrl = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)
    # TODO: Add your code here
    # This seems to work very slowly when mujoco is on, it's much faster when you close mujoco.
    max_area = 0.
    data.actuator("turn 1").ctrl = 1
    for _ in range(1000):
        img = get_image()
        area, _, _ = estimate_pos_of_the_blue_object(img)
        max_area = max(area, max_area)
        # viewer.sync()
        mujoco.mj_step(model, data)

    if max_area < 2000:
        return True

    return False


def find_the_car(img):
    lower = np.array([0, 0, 172])
    upper = np.array([180, 255, 255])

    # Create a mask for red color
    mask = cv2.inRange(img, lower, upper)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # finding contour with maximum area and store it as best_cnt
    max_area = 0.
    best_cnt = None
    for c in contours:
        # (323, 389) is one of the central pixels of the car with the camera
        M = cv2.moments(c)
        if M['m00'] != 0:
            c_x, c_y = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        else:
            c_x, c_y = 323, 389

        area = np.linalg.norm(np.array([c_x, c_y]) - np.array([323, 389]))
        if area > max_area:
            max_area = area
            best_cnt = c

    return max_area, best_cnt


def estimate_pos_of_the_car(img, debug=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_area, best_cnt = find_the_car(img_hsv)

    # finding centroids of best_cnt and draw a circle there
    M = cv2.moments(best_cnt)
    if M['m00'] == 0:
        return 0, -1, -1

    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)

    if debug:
        # save the image with the red circle around the largest red object
        cv2.imwrite("artifacts/car_detection.png", img)

    return max_area, cx, cy


def drive_to_ball_1(seed=1337):
    random.seed(seed)
    steps = random.randint(0, 2500)
    data.actuator("forward 2").ctrl = -1
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.actuator("forward 2").ctrl = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)
    # TODO Add your code here
    b_x, b_y = -1, -1
    data.actuator("turn 1").ctrl = -1
    for _ in range(1000):
        # viewer.sync()
        area, b_x, b_y = estimate_pos_of_the_blue_object(get_image())
        mujoco.mj_step(model, data)
        if area > 1000:
            data.actuator("turn 1").ctrl = 0
            break

    # Now the car with the camera can see both the object and the approaching car.
    print(np.array([b_x, b_y]))
    data.actuator("forward 2").ctrl = 1.
    for i in range(10000):
        # viewer.sync()
        _, car_x, car_y = estimate_pos_of_the_car(get_image())
        print(np.array([car_x, car_y]))
        print(np.linalg.norm(np.array([b_x, b_y]) - np.array([car_x, car_y])))
        THRESHOLD = 10  # Needs to be determined
        if np.linalg.norm(np.array([b_x, b_y]) - np.array([car_x, car_y])) < THRESHOLD:
            break
        mujoco.mj_step(model, data)

        # It's almost working. The problem is that I didn't manage to differentiate between the points
        # on the car with the camera and without the camera on time.


def drive_to_ball_2(seed=1337):
    random.seed(seed)
    steps = random.randint(0, 2500)

    data.actuator("turn 2").ctrl = 1
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.actuator("turn 2").ctrl = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)

    # TODO Add your code here


print(check_ball())
