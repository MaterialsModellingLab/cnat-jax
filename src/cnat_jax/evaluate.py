# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import functools
import json

import flax
import flax.jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import pandas as pd
from absl import logging
from flax.training import checkpoints as flax_checkpoints

from cnat_jax import input_pipeline
from cnat_jax import model as cnat_model


def evaluate(workdir: str):
    """
    Evaluate the CNAT model using the provided configuration and work directory.

    Args:
        config (ml_collections.ConfigDict): Configuration for the evaluation.
        workdir (str): Directory to load the model and store logs.
    """
    with open(f"{workdir}/config.json") as f:
        config = ml_collections.ConfigDict(json.load(f))

    # Setup input pipeline
    dataset_info = input_pipeline.get_dataset_info(config.dataset, "test")

    _, ds_test = input_pipeline.get_datasets(config)
    logging.info(ds_test)
    logging.info(dataset_info)

    model_cls = {"CnaT": cnat_model.CnaTransformer}[config.get("model_type", "CnaT")]
    model = model_cls(num_classes=dataset_info["num_classes"], **config.model)

    def init_model():
        return model.init(
            jax.random.PRNGKey(0),
            jnp.ones((config.batch, config.num_atoms, 3)),
            train=False,
        )

    params = jax.device_put(init_model(), device=jax.devices("cpu")[0])["params"]

    if flax_checkpoints.latest_checkpoint(workdir) is None:
        raise ValueError(
            f"No checkpoint found in {workdir}. Please train the model first."
        )
    ckpt = flax_checkpoints.restore_checkpoint(workdir, target={"params": params})
    params = ckpt["params"]

    infer_fn_repl = jax.pmap(functools.partial(model.apply, train=False))
    params_repl = flax.jax_utils.replicate(params)

    all_data = []
    for test_batch in input_pipeline.prefetch(ds_test, config.prefetch):
        logits = infer_fn_repl({"params": params_repl}, test_batch["atoms"])
        preds = jnp.argmax(logits, axis=-1)
        labels = jnp.argmax(test_batch["label"], axis=-1)

        logits = flax.jax_utils.unreplicate(logits)
        preds = flax.jax_utils.unreplicate(preds)
        labels = flax.jax_utils.unreplicate(labels)
        temperatures = flax.jax_utils.unreplicate(test_batch["temperature"])

        for _i, (logit, pred, label, temp) in enumerate(
            zip(logits, preds, labels, temperatures, strict=True)
        ):
            all_data.append(
                {
                    "confidence": jax.device_get(jnp.max(logit)).item(),
                    "pred": jax.device_get(pred).item(),
                    "label": jax.device_get(label).item(),
                    "temperature": jax.device_get(temp).item(),
                }
            )
    df = pd.DataFrame(all_data)
    print(df)

    df.to_parquet(f"{workdir}/eval.parquet", index=False, compression="snappy")
