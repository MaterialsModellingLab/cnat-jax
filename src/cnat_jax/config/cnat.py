# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0


"""
This module contains the configuration for CNAT (Convolutional Neural Architecture Transformer).
"""

from cnat_jax.config import common, model


def get_config(model_dataset: str):
    """Returns a default configuration for CNAT.

    Args:
        model_dataset: model name and dataset name separated by a comma. e.g., "b15,na_dataset".

    Returns:
        A configuration object with default values.
    """
    model_name, dataset_name = model_dataset.split(",")
    config = common.with_dataset(common.get_config(), dataset_name)
    get_model_config = getattr(model, f"get_{model_name}_config")
    config.model = get_model_config()

    return config
