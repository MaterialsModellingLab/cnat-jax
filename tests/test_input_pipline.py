# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from absl.testing import absltest, parameterized

from cnat_jax import input_pipeline
from cnat_jax.config import cnat as cnat_config


class TrainTest(parameterized.TestCase):
    @parameterized.named_parameters(("n=3", 3), ("n=10", 10), ("n=-1", -1))
    def test_get_datasets(self, num_atoms):
        config = cnat_config.get_config("b15,na_dataset")
        config.num_atoms = num_atoms
        ds_train, ds_test = input_pipeline.get_datasets(config)
        info_train = input_pipeline.get_dataset_info(config.dataset, config.pp["train"])
        info_test = input_pipeline.get_dataset_info(config.dataset, config.pp["test"])

        epsilon = 1e-6  # Tolerance for numerical stability

        for data in ds_train.take(50):
            self.assertIn("atoms", data)
            self.assertIn("label", data)
            self.assertEqual(data["atoms"].shape[-1], 3)
            self.assertEqual(data["label"].shape[-1], info_train["num_classes"])
            # Assert data has no NaN or Inf values
            self.assertFalse(tf.reduce_any(tf.math.is_nan(data["atoms"])))
            self.assertFalse(tf.reduce_any(tf.math.is_inf(data["atoms"])))
            # All value should be in the range of [0, 1]
            r = tf.norm(data["atoms"], axis=-1)
            self.assertTrue(tf.reduce_all(r >= -epsilon))
            self.assertTrue(tf.reduce_all(r <= 1 + epsilon))

            # Labels must remain valid one-hot vectors after preprocessing.
            self.assertTrue(
                tf.reduce_all(tf.equal(tf.reduce_sum(data["label"], axis=-1), 1))
            )

        for data in ds_test.take(50):
            self.assertIn("atoms", data)
            self.assertIn("label", data)
            self.assertEqual(data["atoms"].shape[-1], 3)
            self.assertEqual(data["label"].shape[-1], info_test["num_classes"])
            # Assert data has no NaN or Inf values
            self.assertFalse(tf.reduce_any(tf.math.is_nan(data["atoms"])))
            self.assertFalse(tf.reduce_any(tf.math.is_inf(data["atoms"])))

            # All value should be in the range of [0, 1]
            r = tf.norm(data["atoms"], axis=-1)
            self.assertTrue(tf.reduce_all(r >= -epsilon))
            self.assertTrue(tf.reduce_all(r <= 1 + epsilon))

            # Labels must remain valid one-hot vectors after preprocessing.
            self.assertTrue(
                tf.reduce_all(tf.equal(tf.reduce_sum(data["label"], axis=-1), 1))
            )


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")
    absltest.main()
