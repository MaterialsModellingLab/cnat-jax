# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import ml_collections

MODEL_CONFIG = {}

AUG_REG_CONFIG = {}


def _register(get_config):
    """
    Adds reference to the model config into MODEL_CONFIGS and AUGREG_CONFIGS.
    """

    config = get_config().lock()
    name = config.get("model_name")
    if name not in ("testing"):
        aug_reg_name = name.replace("CnaT", "").replace("+", "_")
        AUG_REG_CONFIG[aug_reg_name] = config
    MODEL_CONFIG[name] = config
    return get_config


@_register
def get_testing_config():
    """
    Returns a simple config used for testing.
    """

    config = ml_collections.ConfigDict()
    config.model_name = "testing"
    config.hidden_size = 8
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4 * 2
    config.transformer.num_heads = 2
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    return config


@_register
def get_b15_config():
    """
    Returns a base config
    """

    config = ml_collections.ConfigDict()
    config.model_name = "CnaT-b15"

    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 256 * 4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.classifier = "token"
    config.representation_size = None

    return config
