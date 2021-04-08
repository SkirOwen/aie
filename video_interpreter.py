import numpy as np
import cv2
import time
from time import monotonic as chrono
from reolinkapi import Camera

import threading

from constants import *


# code to open the webcam
def get_video_feed():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = vc.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    vc.release()
    cv2.destroyWindow("preview")


def save_video(time_limit=10):
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    end_time = chrono() + time_limit

    while cap.isOpened() and chrono() < end_time:
        ret, frame = cap.read()
        if ret:
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return


def extract_frames(path_out, frame_rate=1, time_limit=10):
    """

    Parameters
    ----------
    path_out: str,
              path to save the frames
    time_limit: float, optional
                time to record in seconds, default 10
    frame_rate: int or float, optional
                Extract the frame every "frame_rate" seconds, default 1

    Returns
    -------

    """
    count = 0
    vc = cv2.VideoCapture(0)
    success, image = vc.read()
    success = True

    end_time = chrono() + time_limit
    while success and chrono() < end_time:
        vc.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, image = vc.read()
        cv2.imwrite(path_out + "/frame%d.jpg" % count, image)
        count += frame_rate


def non_blocking(IP, name):
    print("calling non-blocking")

    def inner_callback(img):
        cv2.imshow(name, maintain_aspect_ratio_resize(img, width=600))
        print("got the image non-blocking")
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)

    c = Camera(IP, CAMERA_USER, CAMERA_PSWD)
    # t in this case is a thread
    t = c.open_video_stream(callback=inner_callback)

    print(t.is_alive())
    while True:
        if not t.is_alive():
            print("continuing")
            break
        # stop the stream
        # client.stop_stream()


# Reolink api
def blocking(IP, name):
    c = Camera(IP, CAMERA_USER, CAMERA_PSWD)
    # stream in this case is a generator returning an image (in mat format)
    stream = c.open_video_stream()

    # using next()
    # while True:
    #     img = next(stream)
    #     cv2.imshow("name", maintain_aspect_ratio_resize(img, width=600))
    #     print("got the image blocking")
    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         cv2.destroyAllWindows()
    #         exit(1)

    # or using a for loop
    for img in stream:
        cv2.imshow(name, maintain_aspect_ratio_resize(img, width=600))
        # cv2.imshow("name", img)
        # print("got the image blocking")
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)


# Resizes a image and maintains aspect ratio
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
    # blocking(CAMERA_IP)
    # blocking(CAMERA_IP_2)

    threads = []

    for i in range(3):
        t = threading.Thread(target=non_blocking(CAMERA_IP_LST[i], "cam" + str(i+1)))
        t.daemon = True
        threads.append(t)

    for i in range(3):
        threads[i].start()

    for i in range(3):
        threads[i].join()
