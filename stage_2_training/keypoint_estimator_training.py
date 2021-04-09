#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from adabelief_tf import AdaBeliefOptimizer
import random
import os

# limit VRAM usage
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# enable mixed precision training
keras.mixed_precision.set_global_policy('mixed_float16')

# enable XLA compilation (GPU or CPU)
tf.config.optimizer.set_jit(True)

# preserve GPU threads for better performance
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# create keypoint estimator model
input_size = 96
sota = keras.applications.MobileNetV2(
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='avg',
    weights=None)

NAME = 'keypoint_estimator'
regression = layers.Dense(26, dtype=tf.float32, name="regression")(sota.output)
model_keypoint_estimator = keras.Model(sota.input, regression, name=NAME)


class MaskedMSE(keras.losses.Loss):
    def call(self, y_true, y_pred):
        mask = tf.expand_dims(y_true[:, 2::3], -1)
        y_true = tf.reshape(y_true, (-1, 13, 3))[:, :, :2] * mask
        y_pred = tf.reshape(y_pred, (-1, 13, 2)) * mask
        return tf.reduce_mean(keras.losses.MSE(y_true, y_pred))


model_keypoint_estimator.compile(
    optimizer=AdaBeliefOptimizer(rectify=False, print_change_log=False),
    loss=MaskedMSE())

# gather regression data
training_images_path = "./data_regression/training/images"
training_labels_path = "./data_regression/training/labels"
validation_images_path = "./data_regression/validation/images"
validation_labels_path = "./data_regression/validation/labels"
training_images_list = sorted(
    [os.path.join(training_images_path, filename) for filename in os.listdir(training_images_path)])
training_labels_list = sorted(
    [os.path.join(training_labels_path, filename) for filename in os.listdir(training_labels_path)])
validation_images_list = sorted(
    [os.path.join(validation_images_path, filename) for filename in os.listdir(validation_images_path)])
validation_labels_list = sorted(
    [os.path.join(validation_labels_path, filename) for filename in os.listdir(validation_labels_path)])
training_zip_list = list(zip(training_images_list, training_labels_list))
validation_zip_list = list(zip(validation_images_list, validation_labels_list))
random.shuffle(training_zip_list)
random.shuffle(validation_zip_list)
training_list = tf.data.Dataset.from_tensor_slices(training_zip_list)
validation_list = tf.data.Dataset.from_tensor_slices(validation_zip_list)
print("Number of training data:", len(training_list))
print("Number of validation data:", len(validation_list))


# load regression data
def preprocess_data(sample):
    image = tf.io.read_file(sample[0])
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [input_size, input_size], antialias=True)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    label = tf.io.read_file(sample[1])
    label = tf.strings.split(label, sep=', ')
    label = tf.strings.to_number(label, out_type='float32')
    return image, label


# apply data augmentation
def data_augmentation(image, label):
    if tf.equal(tf.random.uniform(shape=(), minval=0, maxval=2, dtype='int32'), 1):
        image = tf.image.flip_left_right(image)
        label = tf.stack(([1.] * 13, label[1::3] * 2, label[2::3] * 2), axis=1) - tf.reshape(label, (13, 3))
        label = tf.reshape(
            tf.gather(label, [0] + [val for pair in zip(range(2, 13, 2), range(1, 12, 2)) for val in pair]), (39,))
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_hue(image, 0.1)
    image = tf.clip_by_value(image, -1., 1.)
    return image, label


# create regression dataset
BATCH_SIZE = 32
training_data = training_list.map(
    preprocess_data,
    num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(len(training_list)).map(
    data_augmentation,
    num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
validation_data = validation_list.map(
    preprocess_data,
    num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# train keypoint estimator
EPOCHS = 500
weights_path = "./weights/keypoint_estimator/ckpt"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
history = model_keypoint_estimator.fit(
    training_data,
    epochs=EPOCHS,
    validation_data=validation_data,
    callbacks=[model_checkpoint_callback],
    verbose=2)

# load best weights and evaluate on validation data
model_keypoint_estimator.load_weights(weights_path)
print("Best weights loaded")
results = model_keypoint_estimator.evaluate(validation_data)
print("Evaluation results:", results)

# save keypoint estimator model
model_path = "./models/keypoint_estimator"
model_keypoint_estimator.save(model_path, include_optimizer=False)

# plot history
fig = plt.figure()
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss (Best: %.5f)' % results)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
results_save_path = "./keypoint_estimator_loss_"+str(BATCH_SIZE)+".png"
fig.savefig(results_save_path, dpi=192)
print("Results saved to", results_save_path)
