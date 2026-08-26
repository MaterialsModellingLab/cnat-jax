# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import ml_collections
import tensorflow as tf
from absl.testing import absltest, parameterized

from cnat_jax import train
from cnat_jax.config import common, model


class TrainTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("tfds", "tfds"),
        # ("directory", "directory"),
    )
    def test_train_and_evaluate(self, dataset_source):
        config = common.get_config()
        config.model = model.get_testing_config()
        config.batch = 64
        config.accum_steps = 2
        config.batch_eval = 8
        config.total_steps = 1
        config.num_atoms = 10

        with tempfile.TemporaryDirectory() as workdir:
            if dataset_source == "tfds":
                config.dataset = "na_dataset"
                config.pp = ml_collections.ConfigDict(
                    {
                        "train": "train[:98%]",
                        "test": "test",
                    }
                )
            else:
                raise ValueError(f"Unknown dataset source: {dataset_source}")

            _ = train.train(config, workdir)
            self.assertTrue(os.path.exists(f"{workdir}/checkpoint_1"))


if __name__ == "__main__":
    # To prevent out-of-memory errors during testing
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    tf.config.experimental.set_visible_devices([], "GPU")
    absltest.main()
