# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0


import logging as python_logging
import threading
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import logging
from tensorflow_graphics.geometry.transformation import quaternion


class GFileHandler(python_logging.StreamHandler):
    """
    A file handler that uses gfile to write to a file.
    """

    def __init__(self, filename, mode, flush_secs=1.0):
        super().__init__()
        tf.io.gfile.makedirs(Path(filename).parent)
        if mode == "a" and not tf.io.gfile.exists(filename):
            mode = "w"
        self.file_handle = tf.io.gfile.GFile(filename, mode)
        self.flush_secs = flush_secs
        self.flush_timer = None

    def flush(self):
        self.file_handle.flush()

    def emit(self, record):
        msg = self.format(record=record)
        self.file_handle.write(f"{msg}\n")
        if self.flush_timer is not None:
            self.flush_timer.cancel()
        self.flush_timer = threading.Timer(self.flush_secs, self.flush)
        self.flush_timer.start()


def add_gfile_logger(
    workdir: str, *, basename: str = "train", level=python_logging.INFO
):
    """Add a gfile logger to the root logger."""
    file_handler = GFileHandler(f"{workdir}/{basename}.log", "a")
    file_handler.setLevel(level=level)
    file_handler.setFormatter(logging.PythonFormatter())
    python_logging.getLogger("").addHandler(file_handler)


def create_learning_rate_schedule(
    total_steps: int,
    base_lr: float,
    decay_type: str,
    warmup_steps: int,
    linear_end: float = 1e-5,
) -> Callable:
    """
    Create learning rate schedule.

    Args:
        total_steps: Total number of steps.
        base_lr: Base learning rate.
        decay_type: Type of decay. Can be "linear" or "cosine".
        warmup_steps: Number of warmup steps.
        linear_end: End learning rate for linear decay.
    """

    def step_fn(step) -> jnp.ndarray:
        lr = base_lr

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        if decay_type == "linear":
            lr = linear_end + (lr - linear_end) * (1.0 - progress)
        elif decay_type == "cosine":
            lr = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")
        if warmup_steps:
            lr = lr * jnp.minimum(1.0, step / float(warmup_steps))
        return jnp.asarray(lr, dtype=jnp.float32)

    return step_fn


def accumulate_gradient(loss_and_grad_fn, params, inputs, labels, accum_steps):
    """
    Accumulate gradients over multiple steps.

    Args:
        loss_and_grad_fn: Function to compute loss and gradients.
        params: Model parameters.
        inputs: Input data.
        labels: Labels.
        accum_steps: Number of steps to accumulate gradients.

    Returns:
        Loss and gradients.
    """
    if accum_steps and accum_steps > 1:
        assert inputs.shape[0] % accum_steps == 0, (
            "Batch size must be divisible by accum_steps"
        )
        step_size = inputs.shape[0] // accum_steps
        loss, grad = loss_and_grad_fn(params, inputs[:step_size], labels[:step_size])

        def acc_grad_and_loss(i, loss_and_grad):
            input_partial = jax.lax.dynamic_slice(
                inputs, (i * step_size, 0, 0), (step_size, *inputs.shape[1:])
            )
            label_partial = jax.lax.dynamic_slice(
                labels, (i * step_size, 0), (step_size, labels.shape[1])
            )
            loss_i, grad_i = loss_and_grad_fn(params, input_partial, label_partial)
            loss, grad = loss_and_grad
            return (loss + loss_i, jax.tree.map(lambda x, y: x + y, grad, grad_i))

        loss, grad = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (loss, grad))
        return jax.tree.map(lambda x: x / accum_steps, (loss, grad))
    else:
        return loss_and_grad_fn(params, inputs, labels)


def random_rot(input: tf.Tensor) -> tf.Tensor:
    """
    Apply a random rotation to the input tensor.

    Args:
        input_tensor: Input tensor of shape (N, 3).

    Returns:
        Rotated tensor of the same shape.
    """

    x = tf.random.uniform(shape=(3,), minval=0.0, maxval=1.0, dtype=tf.float32)
    r1 = tf.sqrt(1.0 - x[0])
    r2 = tf.sqrt(x[0])
    theta1 = 2.0 * jnp.pi * x[1]
    theta2 = 2.0 * jnp.pi * x[2]
    s1 = tf.sin(theta1)
    c1 = tf.cos(theta1)
    s2 = tf.sin(theta2)
    c2 = tf.cos(theta2)
    random_quaternion = tf.stack([r1 * s1, r1 * c1, r2 * s2, r2 * c2], axis=-1)
    return quaternion.rotate(input, random_quaternion)
