import constants
import video_interpreter as vi

import numpy as np
import tensorflow as tf


def load_coco():
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
    pass


def n2a_dataset():
    pass


def preprocess_data(sample):
    image = tf.io.read_file(sample[0])
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [299, 299], antialias=True)
    image = tf.keras.applications.xception.preprocess_input(image)

    label = sample[1]
    label = tf.strings.to_number(label, out_type='int32')

    return image, label


def data_augmentation(image, label):
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_hue(image, 0.1)
    image = tf.clip_by_value(image, -1.0, 1.0)

    return image, label

