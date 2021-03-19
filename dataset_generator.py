import os
import sys
import pickle       # see if useful


import numpy as np
import cv2


PATH_DATASET = "./dataset"

RUNNING_PATH = os.path.join(PATH_DATASET, "running")
CYCLING_PATH = os.path.join(PATH_DATASET, "cycling")
SITTING_PATH = os.path.join(PATH_DATASET, "sitting")
WALKING_PATH = os.path.join(PATH_DATASET, "walking")
# walking should the default classifier, but creating the folder here just in case


# code to open the webcam
# TODO: test with a webcam..  dont have one this computer at the moment lol
# TODO: time limit for the video capture
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
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # When everything done, release the capture
    vc.release()
    cv2.destroyWindow("preview")


def save_video():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame, 0)

            # write the flipped frame
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


if __name__ == '__main__':
    pass

