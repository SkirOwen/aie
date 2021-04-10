import os

CAMERA_IP_1 = "192.168.1.234"
CAMERA_IP_2 = "192.168.1.241"
CAMERA_IP_3 = "192.168.1.242"

CAMERA_IP_LST = [CAMERA_IP_1, CAMERA_IP_2, CAMERA_IP_3]

CAMERA_USER = "admin"
CAMERA_PSWD = "camera"

DATASET_DIR = "./dataset"

TRAINING_DIR = os.path.join(DATASET_DIR, "training")

RUNNING_DIR = os.path.join(TRAINING_DIR, "running")
CYCLING_DIR = os.path.join(TRAINING_DIR, "cycling")
SITTING_DIR = os.path.join(TRAINING_DIR, "sitting")
WALKING_DIR = os.path.join(TRAINING_DIR, "walking")
# walking should the default classifier, but creating the folder here just in case


# TODO: ask if important two different folders for training and testing?
def initialize_directories():
    for folder in [RUNNING_DIR, CYCLING_DIR, SITTING_DIR, WALKING_DIR]:
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder)

