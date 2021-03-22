import os
import sys
import pickle       # see if useful
import numpy as np
import cv2

import constants
import video_interpreter as vi
import to_nodes


def generate_ctm_dataset(activity):
    # func to extract frames correctly here, should return either frames by frames
    # or multiple frames in a convenient data format (more realistic) after a certain
    # length of recording (e.g. 30 sec video)

    # nodes_extraceted = to_nodes.f2n(*frames)    # maybe map(to_nodes.f2n(), frames)??

    # save in certain way node_extracted (maybe trough func)
    pass


if __name__ == '__main__':
    constants.initialize_directories()

