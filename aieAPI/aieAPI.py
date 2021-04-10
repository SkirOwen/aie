#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from tensorflow.python.saved_model import tag_constants


class AIE:
    """create an instance that provides inference of stage_1 and stage_2 models"""

    def __init__(self):
        print("Initializing...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        self.input_size = 416
        self.target_size = 96
        self.id_to_label = ('cycling', 'jogging', 'sitting', 'other')
        self.id_to_color = ((255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128))
        self.skeleton = (
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (8, 10), (9, 11),
            (10, 12))
        self.edge_colors = ((255, 0, 0),
                            (255, 135, 0),
                            (255, 211, 0),
                            (222, 255, 10),
                            (161, 255, 10),
                            (10, 255, 153),
                            (10, 239, 255),
                            (20, 125, 245),
                            (88, 10, 255),
                            (239, 195, 5),
                            (116, 255, 82),
                            (15, 190, 199),
                            (105, 68, 250),
                            (239, 195, 5))
        print("Loading models...")
        self.model_human_detector = tf.saved_model.load("./aieAPI/human_detector", tags=[tag_constants.SERVING])
        self.infer_human_detector = self.model_human_detector.signatures['serving_default']
        self.model_classifier = tf.saved_model.load("./aieAPI/classifier", tags=[tag_constants.SERVING])
        self.infer_classifier = self.model_classifier.signatures['serving_default']
        self.model_keypoint_estimator = tf.saved_model.load("./aieAPI/keypoint_estimator", tags=[tag_constants.SERVING])
        self.infer_keypoint_estimator = self.model_keypoint_estimator.signatures['serving_default']
        print("Allocating memory...")
        _ = self.detect(
            cv2.cvtColor(cv2.imread("./aieAPI/warm_up/warm_up.jpg"), cv2.COLOR_BGR2RGB),
            show_bbox=False, show_skeleton=False, show_activity=False)
        print("Ready√")

    def get_activity(self, logits):
        activity_id = np.argmax(logits)
        if logits[activity_id] < 0.6:
            activity_id = 3
        return activity_id, self.id_to_label[activity_id], self.id_to_color[activity_id]

    def detect(self, original_image, show_bbox=True, show_skeleton=True, show_activity=True):
        """
        Parameters:
                original_image: RGB image (numpy array) of any size, dtype=uint8 ~ [0, 255]
                show_bbox: whether to draw bounding boxes on targets
                show_skeleton: whether to draw skeletons on targets
                show_activity: whether to display activity labels on targets
        
        Returns:
                output_image: RGB image (numpy array) with skeletons and activity labels, dtype=uint8 ~ [0, 255]
                count of activities: [<cycling>, <jogging>, <sitting>, <unknown>]
        """
        output_image = original_image.copy()
        H, W, _ = output_image.shape
        resized_image = cv2.resize(output_image, (self.input_size, self.input_size), cv2.INTER_CUBIC) / 255.
        resized_image = tf.expand_dims(resized_image.astype(np.float32), axis=0)

        pred_bbox = self.infer_human_detector(resized_image)['tf.concat_12'].numpy().reshape(-1, 84)
        boxes = pred_bbox[:, 0:4]
        scores = pred_bbox[:, 4:]
        human_indices = np.argwhere(np.argmax(scores, axis=-1) == 0)
        boxes = boxes[human_indices, :].reshape(-1, 4)
        scores = scores[human_indices, 0].reshape(-1)
        selected_indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=20,
            iou_threshold=0.45,
            score_threshold=0.25)
        boxes = np.rint(np.clip(boxes[selected_indices.numpy(), :], 0., 1.) * [H, W, H, W]).astype(np.int32)

        activity_count = [0, 0, 0, 0]
        for y_min, x_min, y_max, x_max in boxes:
            line_px = min(5, max(1, round(100 * (y_max - y_min) * (x_max - x_min) / H / W)))
            target = keras.applications.mobilenet_v2.preprocess_input(tf.expand_dims(cv2.resize(
                output_image[y_min:y_max, x_min:x_max, :], (self.target_size, self.target_size),
                cv2.INTER_CUBIC).astype(np.float32), axis=0))
            keypoints = self.infer_keypoint_estimator(target)['regression'].numpy().reshape(-1, 2)
            activity_id, activity, label_color = self.get_activity(
                np.squeeze(self.infer_classifier(target)['classification'].numpy()))

            activity_count[activity_id] += 1
            Xs = np.rint(keypoints[:, 0] * (x_max - x_min) + x_min).astype(np.int32)
            Ys = np.rint(keypoints[:, 1] * (y_max - y_min) + y_min).astype(np.int32)

            if show_bbox:
                _ = cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), label_color, line_px)

            if show_skeleton:
                for index, edge_color in zip(self.skeleton, self.edge_colors):
                    _ = cv2.line(output_image, (Xs[index[0]], Ys[index[0]]), (Xs[index[1]], Ys[index[1]]), edge_color,
                                 line_px, cv2.LINE_AA)

            if show_activity:
                text_x, text_y = (Xs[0], Ys[0])
                (text_w, text_h), text_bl = cv2.getTextSize(activity, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                _ = cv2.rectangle(output_image, (text_x, text_y - text_h), (text_x + text_w, text_y + text_bl),
                                  (0, 0, 0), -1)
                _ = cv2.putText(output_image, activity, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1,
                                cv2.LINE_AA)

        return output_image, activity_count


def call_aie(original_frame, fps):
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    output_frame, _ = AIE.detect(original_frame, show_bbox=False, show_skeleton=True, show_activity=True)
    _ = cv2.putText(output_frame, "FPS: %.1f" % fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
