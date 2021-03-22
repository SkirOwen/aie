import numpy as np
import cv2


# code to open the webcam
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    vc.release()
    cv2.destroyWindow("preview")


def save_video():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
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


def extract_frames(path_out, frame_rate=1):
    """

    Parameters
    ----------
    frame_rate: int or float,
                Extract the frame every "frame_rate" seconds

    Returns
    -------

    """
    count = 0
    vc = cv2.VideoCapture(0)
    success, image = vc.read()
    success = True
    while success:
        vc.set(cv2.CAP_PROP_POS_MSEC,(count * 1000))
        success, image = vc.read()
        cv2.imwrite(path_out + "/frame%d.jpg" % count, image)
        count += frame_rate