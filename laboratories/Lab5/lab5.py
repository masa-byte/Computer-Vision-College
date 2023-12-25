import cv2
from cv2 import aruco
import numpy as np
import os

markerLength = 3.42  # cm
markerSeparation = 0.67  # cm


def load_images():
    folder = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(folder, "Aruco")
    images = []
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    return images


def load_video():
    folder = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(folder, "Aruco")
    video = cv2.VideoCapture(os.path.join(folder, "Aruco_board.mp4"))
    return video


def calibrate_camera(images):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
    image_shape = images[0].shape

    corners_list = []
    ids_list = []
    counter = []
    first_done = False
    image_shape = images[0].shape
    arucoParams = aruco.DetectorParameters_create()
    for image in images:
        corners, ids, rejected_corners = aruco.detectMarkers(
            image, aruco_dict, parameters=arucoParams
        )
        if first_done:
            corners_list = np.vstack((corners_list, corners))
            ids_list = np.vstack((ids_list, ids))
        else:
            corners_list = corners
            ids_list = ids
            first_done = True
        counter.append(len(ids))

    counter = np.array(counter)
    camera_matrix_init = np.array([
        [image_shape[1], 0, image_shape[1] / 2],
        [0, image_shape[1], image_shape[0] / 2],
        [0, 0, 1],
    ])

    dist_coeffs_init = np.zeros((5, 1))

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraAruco(
        corners_list, ids_list, counter, board, image_shape, camera_matrix_init, dist_coeffs_init
    )

    print("Camera matrix: ", camera_matrix)
    print("Distortion coefficients: ", dist_coeffs)

    return camera_matrix, dist_coeffs


def check_camera_calibration(camera_matrix, dist_coeffs):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
    arucoParams = aruco.DetectorParameters_create()

    video = load_video()
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_shape[:2], 1, image_shape[:2]
    )

    while True:
        ret, frame = video.read()

        if not ret:
            break

        img_aruco = frame.copy()
        img_undistorted = cv2.undistort(
            img_aruco, camera_matrix, dist_coeffs, None, new_mtx
        )
        corners, ids, rejected_corners = aruco.detectMarkers(
            img_undistorted, aruco_dict, parameters=arucoParams
        )

        if ids is not None:
            ret, rvec, tvec = aruco.estimatePoseBoard(
                corners, ids, board, new_mtx, dist_coeffs, None, None
            )
            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img_undistorted, corners, ids, (0, 255, 0))
                img_aruco = aruco.drawAxis(
                    img_aruco, camera_matrix, dist_coeffs, rvec, tvec, 10
                )
        cv2.imshow("frame", img_aruco)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    images = load_images()
    image_shape = images[0].shape
    camera_matrix, dist_coeffs = calibrate_camera(images)
    check_camera_calibration(camera_matrix, dist_coeffs)
