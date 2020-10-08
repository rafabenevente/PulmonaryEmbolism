import os
from typing import Callable, Tuple, List

import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from imgaug import augmenters as iaa


import pre_process


def rescale(images: tf.Tensor) -> tf.Tensor:
    return images / 255.0


class ImageDataLoader(object):
    def __init__(self,
                 df: pd.DataFrame,
                 images_dir_path: str,
                 image_size: Tuple[int, int],
                 image_color_mode: str = "rgb",
                 dtype: str = "float32",
                 image_filename_column: str = "fileName",
                 label_column: List[str] = ["label"],
                 cache: bool = False,
                 image_preprocessing_function: Callable[[tf.Tensor], tf.Tensor] = rescale,
                 augmenter: iaa.Augmenter = None) -> None:

        super().__init__()

        self._image_paths = np.array(
            [
                os.path.join(images_dir_path, image)
                for image in df[image_filename_column]
            ]
        )
        self._labels = df[label_column].to_dict("list")

        self._image_size = image_size
        self._image_color_mode = image_color_mode
        self._dtype = dtype
        self._cache = cache
        self._image_preprocessing_function = image_preprocessing_function
        self._augmenter = augmenter

    @property
    def _image_shape(self) -> Tuple[int, int, int]:
        return (
            self._image_size[0],
            self._image_size[0],
            3 if self._image_color_mode == "rgb" else 1,
        )

    def _tf_load_image(self, image_path: tf.Tensor, label: tf.Tensor) -> tf.data.Dataset:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(
            image, channels=3 if self._image_color_mode == "rgb" else 1
        )
        image = tf.reshape(image, shape=self._image_shape)

        return tf.data.Dataset.from_tensors((image, label))

    def _tf_augment_and_preprocess(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img_shape = tf.shape(images)
        if self._augmenter:
            img_dtype = images.dtype
            images = tf.numpy_function(
                self._augmenter.augment_images, [images], img_dtype
            )
            images = tf.reshape(images, shape=img_shape)

        images = tf.cast(images, tf.as_dtype(self._dtype))
        images = self._image_preprocessing_function(images)

        return images, labels

    def to_tf_dataset(self, batch_size: int, seed: int = 42) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((self._image_paths, self._labels))
        for img, lab in dataset:
            self._tf_load_image(img, lab)
        dataset = dataset.interleave(
            self._tf_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if self._cache:
            dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=batch_size)

        dataset = dataset.map(
            self._tf_augment_and_preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
