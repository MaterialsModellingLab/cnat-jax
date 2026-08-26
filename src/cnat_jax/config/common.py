# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import ml_collections


def get_config():
    """Returns a default configuration."""
    config = ml_collections.ConfigDict()

    config.pretrained_dir = "."
    config.dataset = None
    config.seed = 0

    # Number of atoms to use in the dataset.
    config.num_atoms = 345

    # Path to manually downloaded dataset
    config.tfds_manual_dir = None
    # Path to tensorflow datasets directory
    config.tfds_data_dir = None
    # Number of steps
    config.total_steps = None

    # Resizes global gradients
    config.grad_norm_clip = 1.0
    # Datatype to use for momentum state (bfloat16, float32)
    config.optim_dtype = "bfloat16"
    # Accumulate gradients over multiple steps to save memory
    config.accum_steps = 8

    # Batch size for training
    config.batch = 512
    # Batch size for evaluation
    config.batch_eval = 64  # Test size is 4800, 4800 % config.batch_eval should be 0
    # Shuffle buffer size
    config.shuffle_buffer = 50_000
    # Run prediction on validation set every so many steps
    config.eval_every = 100
    # Log progress every so many steps
    config.progress_every = 10
    # How often to write checkpoints. Specifying 0 disables checkpointing.
    config.checkpoint_every = 1_000

    # Number of batches to prefetch to device
    config.prefetch = 2

    # Base learning rate
    config.base_lr = 1e-2
    # How to decay the learning rate ("cosine", "linear")
    config.decay_type = "cosine"
    # Number of warmup steps for learning rate
    config.warmup_steps = 500

    # Wil be set from model.py
    config.model = None
    # Must be set via `with_dataset`
    config.pp = None

    return config.lock()


DATASET_PRESETS = {
    "na_dataset": ml_collections.ConfigDict(
        {
            "total_steps": 20_000,
            "pp": ml_collections.ConfigDict(
                {
                    "train": "train",
                    "test": "test",
                }
            ),
        }
    ),
}


def with_dataset(
    config: ml_collections.ConfigDict, dataset: str
) -> ml_collections.ConfigDict:
    """Returns a configuration with the specified dataset."""
    config = ml_collections.ConfigDict(config.to_dict())
    config.dataset = dataset
    config.update(DATASET_PRESETS[dataset])
    return config
