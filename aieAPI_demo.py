#! /usr/bin/env python
# coding=utf-8

import cv2
from aieAPI.aieAPI import AIE
from time import perf_counter

from constants import *


def test_on_image(image_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    output_image, activity_count = aie.detect(original_image, show_bbox=False, show_skeleton=True, show_activity=True)
    print("Activity count:", activity_count)
    cv2.imshow('output image', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = 0.
    while cap.isOpened():
        tic = perf_counter()
        ret, original_frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        output_frame, _ = aie.detect(original_frame, show_bbox=False, show_skeleton=True, show_activity=True)
        _ = cv2.putText(output_frame, "FPS: %.1f" % fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        cv2.imshow('frame', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        fps = 1. / (perf_counter() - tic)
    cap.release()
    cv2.destroyAllWindows()


def test_on_webcam():
    cap = cv2.VideoCapture(0)
    fps = 0.
    while cap.isOpened():
        tic = perf_counter()
        ret, original_frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        output_frame, _ = aie.detect(original_frame, show_bbox=False, show_skeleton=True, show_activity=True)
        _ = cv2.putText(output_frame, "FPS: %.1f" % fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        cv2.imshow('frame', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        fps = 1. / (perf_counter() - tic)
    cap.release()
    cv2.destroyAllWindows()


def test_on_camera(link):
    cap = cv2.VideoCapture(link)
    fps = 0.
    while cap.isOpened():
        tic = perf_counter()
        ret, original_frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        output_frame, _ = aie.detect(original_frame, show_bbox=False, show_skeleton=True, show_activity=True)
        _ = cv2.putText(output_frame, "FPS: %.1f" % fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        cv2.imshow('frame', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR),  maintain_aspect_ratio_resize(output_frame, width=600))
        fps = 1. / (perf_counter() - tic)
    cap.release()
    cv2.destroyAllWindows()


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    # if width is None and height is None:
    #     return image

    # We are resizing height if width is none
    if width is None:
        if height is None:
            return image
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    # else:
    # Calculate the ratio of the 0idth and construct the dimensions
    r = width / float(w)
    dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


if __name__ == '__main__':
    aie = AIE()

    camera0 = 'rtsp://admin:camera@192.168.1.234:554/h264Preview_01_main'.format(CAMERA_USER, CAMERA_PSWD)
    camera1 = 'rtsp://admin:camera@192.168.1.241:554/h264Preview_01_main'.format(CAMERA_USER, CAMERA_PSWD)
    camera2 = 'rtsp://admin:camera@192.168.1.242:554/h264Preview_01_main'.format(CAMERA_USER, CAMERA_PSWD)

    # test_on_image("./test_images/cycling.jpg")
    # test_on_video("./test_videos/jogging_3.mp4")
    # test_on_webcam()
    test_on_camera(camera1)
