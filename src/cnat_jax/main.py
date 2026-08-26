# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import jax
import tensorflow as tf
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

from cnat_jax import evaluate, train, util

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory to store logs and model data.")
flags.DEFINE_enum(
    "mode",
    None,
    ["train", "evaluate"],
    "Mode of operation: 'train' to train the model, 'evaluate' to evaluate the model.",
)
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)
flags.mark_flag_as_required("mode")
flags.mark_flag_as_required("workdir")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    util.add_gfile_logger(FLAGS.workdir)

    # Set up TensorFlow to not use GPU
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    jax_xla_backend = "None" if FLAGS.jax_xla_backend == "" else FLAGS.jax_xla_backend
    logging.info("JAX XLA backend: %s", jax_xla_backend)

    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    if FLAGS.mode == "train":
        flags.mark_flag_as_required("config")
        train.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "evaluate":
        evaluate.evaluate(FLAGS.workdir)
    else:
        raise app.UsageError(f"Unknown mode: {FLAGS.config.mode}")


def run():
    jax.config.config_with_absl()
    app.run(main)


if __name__ == "__main__":
    run()
