import os

DATASET_DIR = "./dataset"

RUNNING_DIR = os.path.join(DATASET_DIR, "running")
CYCLING_DIR = os.path.join(DATASET_DIR, "cycling")
SITTING_DIR = os.path.join(DATASET_DIR, "sitting")
WALKING_DIR = os.path.join(DATASET_DIR, "walking")
# walking should the default classifier, but creating the folder here just in case


# TODO: ask if important two different folders for training and testing?
def initialize_directories():
    for folder in [RUNNING_DIR, CYCLING_DIR, SITTING_DIR, WALKING_DIR]:
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder)

