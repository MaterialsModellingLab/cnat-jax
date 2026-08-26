# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0
"""
input_pipeline.py

Input pipeline for CNAT-JAX.
This module provides functions to load and preprocess datasets for training and evaluation.
It supports loading datasets from TensorFlow Datasets (TFDS).
"""

import flax
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

from cnat_jax.util import random_rot


def get_dataset_info(dataset: str, split):
    """
    Returns dataset information based on the dataset name and split.

    Args:
        dataset (str): The name of the dataset.
        split (str): The split of the dataset ('test', 'train').
    """
    # TODO(anyone): Support for loading local datasets

    # TFDS dataset info loading
    return get_tfds_info(dataset, split)


def get_tfds_info(dataset, split):
    """Returns dataset information based on the dataset name and split.

    Args:
        dataset (str): The name of the dataset.
        split (str): The split of the dataset ('test', 'train').
    Returns:
        A dictionary with the following keys:
        - num_examples: The number of examples in the dataset.
        - num_classes: The number of classes in the dataset.
        - int2str: A function that converts an integer to a class name.
    """
    builder = tfds.builder(dataset)
    return {
        "num_examples": builder.info.splits[split].num_examples,
        "num_classes": builder.info.features["label"].num_classes,
        "int2str": builder.info.features["label"].int2str,
    }


def get_datasets(config):
    """Returns the datasets for training and evaluation.

    Args:
        config: The configuration object containing dataset information.
    Returns:
        A tuple of (train_dataset, test_dataset).
    """
    # Set seed
    tf.random.set_seed(config.seed)

    # TODO(anyone): Support for loading local datasets

    # TFDS datasets loading
    logging.info('Loading datasets from tfds "%s"', config.dataset)
    ds_train = get_data_from_tfds(config=config, mode="train")
    ds_test = get_data_from_tfds(config=config, mode="test")
    return ds_train, ds_test


def get_data_from_tfds(*, config, mode):
    """Returns a dataset from TensorFlow Datasets (TFDS).

    Args:
        config: The configuration object containing dataset information.
        mode: The mode of the dataset ('test', 'train').
    Returns:
        A dataset object.
    """
    builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=config.tfds_manual_dir)
    )

    data = builder.as_dataset(
        split=config.pp[mode],
        shuffle_files=(mode == "train"),
    )

    dataset_info = get_tfds_info(config.dataset, config.pp[mode])

    return get_data(
        data=data,
        mode=mode,
        num_classes=dataset_info["num_classes"],
        repeats=None if mode == "train" else 1,
        batch_size=config.batch_eval if mode == "test" else config.batch,
        shuffle_buffer=min(dataset_info["num_examples"], config.shuffle_buffer),
        num_atoms=config.num_atoms,
    )


def get_data(
    *,
    data: tf.data.Dataset,
    mode: str,
    num_classes: int,
    repeats: int,
    batch_size: int,
    shuffle_buffer: int,
    num_atoms: int,
    preprocess=None,
):
    """Returns a dataset with preprocessing applied.
    Args:
        data: The dataset to preprocess.
        mode: The mode of the dataset ('test', 'train').
        num_classes: The number of classes in the dataset.
        repeats: The number of times to repeat the dataset.
        batch_size: The batch size for the dataset.
        shuffle_buffer: The buffer size for shuffling the dataset.
        preprocess: A function to preprocess the data.
    Returns:
        A preprocessed dataset object.
    """

    def _pp(data):
        atoms = data["atoms"]

        # Rotate randomly
        atoms = random_rot(atoms)

        # Select only the first num_atoms atoms
        atoms = atoms[:num_atoms, :]

        # Normalize atoms
        atoms = atoms / tf.reduce_max(tf.norm(atoms, axis=-1))

        label = tf.one_hot(data["label"], num_classes)
        return {"atoms": atoms, "label": label, "temperature": data["temperature"]}

    data = data.repeat(repeats)
    if mode == "train":
        data = data.shuffle(shuffle_buffer)

    if preprocess is not None:
        data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
    data = data.map(_pp, tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)

    # Shard data such that it can be distributed across devices
    num_devices = jax.local_device_count()

    def _shard(data):
        data["atoms"] = tf.reshape(
            data["atoms"], [num_devices, -1, *data["atoms"].shape[1:]]
        )
        data["label"] = tf.reshape(data["label"], [num_devices, -1, num_classes])
        data["temperature"] = tf.reshape(
            data["temperature"], [num_devices, -1, *data["temperature"].shape[1:]]
        )
        return data

    if num_devices is not None:
        data = data.map(_shard, tf.data.experimental.AUTOTUNE)

    return data.prefetch(1)


def prefetch(dataset, n_prefetch: int):
    """Prefetches the dataset for faster training.

    Args:
        dataset: The dataset to prefetch.
        n_prefetch: The number of batches to prefetch.
    Returns:
        A prefetch dataset object.
    """
    ds_iter = iter(dataset)
    ds_iter = (jax.tree.map(lambda t: np.asarray(memoryview(t)), x) for x in ds_iter)
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter
