# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import functools
import time

import flax
import flax.jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from clu import metric_writers, periodic_actions
from flax.training import checkpoints as flax_checkpoints

from cnat_jax import input_pipeline, util
from cnat_jax import model as cnat_model


def make_update_fn(*, apply_fn, accum_steps, tx):
    """
    Create a function to update the model parameters for data parallel training.
    """

    def update_fn(params, opt_state, batch, rng):
        _, new_rng = jax.random.split(rng)
        # Bind the rng key to the device id (which is unique across hosts)
        dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

        def cross_entropy_loss(*, logits, labels):
            logp = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.sum(logp * labels, axis=1))

        def loss_fn(params, inputs, labels):
            logits = apply_fn(
                {"params": params},
                rngs={"dropout": dropout_rng},
                inputs=inputs,
                train=True,
            )
            return cross_entropy_loss(logits=logits, labels=labels)

        loss, grad = util.accumulate_gradient(
            jax.value_and_grad(loss_fn),
            params,
            batch["atoms"],
            batch["label"],
            accum_steps,
        )
        grad = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="batch"), grad)
        updates, opt_state = tx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        loss = jax.lax.pmean(loss, axis_name="batch")

        return params, opt_state, loss, new_rng

    return jax.pmap(update_fn, axis_name="batch", donate_argnums=(0,))


def train(config: ml_collections.ConfigDict, workdir: str):
    """Train and evaluate the model."""
    # Save config to workdir
    with open(f"{workdir}/config.json", "w") as f:
        import json

        json.dump(config.to_dict(), f, indent=2)

    # Setup Key
    key = jax.random.PRNGKey(config.seed)
    _, init_key, model_key = jax.random.split(key, 3)

    # Setup input pipeline
    dataset_info = input_pipeline.get_dataset_info(config.dataset, "train")

    ds_train, ds_test = input_pipeline.get_datasets(config)
    logging.info(ds_train)
    logging.info(ds_test)

    # Building Transformer architecture
    model_cls = {"CnaT": cnat_model.CnaTransformer}[config.get("model_type", "CnaT")]

    model = model_cls(num_classes=dataset_info["num_classes"], **config.model)

    def init_model():
        return model.init(
            init_key,
            jnp.ones((config.batch, config.num_atoms, 3)),
            train=False,
        )

    params = jax.device_put(init_model(), device=jax.devices("cpu")[0])["params"]

    initial_step = 1
    total_steps = config.total_steps

    lr_fn = util.create_learning_rate_schedule(
        total_steps=total_steps,
        base_lr=config.base_lr,
        decay_type=config.decay_type,
        warmup_steps=config.warmup_steps,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_norm_clip),
        optax.sgd(
            learning_rate=lr_fn,
            momentum=0.9,
            accumulator_dtype=config.optim_dtype,
        ),
    )

    opt_state = tx.init(params)

    ckpt = flax_checkpoints.restore_checkpoint(
        workdir, {"params": params, "opt_state": opt_state, "step": initial_step}
    )
    params, opt_state, initial_step = ckpt["params"], ckpt["opt_state"], ckpt["step"]
    logging.info("Will start/continue training at initial_step=%d", initial_step)
    params_repl, opt_state_repl = flax.jax_utils.replicate((params, opt_state))

    update_fn_repl = make_update_fn(
        apply_fn=model.apply, accum_steps=config.accum_steps, tx=tx
    )
    infer_fn_repl = jax.pmap(functools.partial(model.apply, train=False))
    update_rng_repl = flax.jax_utils.replicate(model_key)

    # Setup metric writer & hooks
    writer = metric_writers.create_default_writer(workdir, asynchronous=False)
    hooks = [
        periodic_actions.Profile(logdir=workdir),
        periodic_actions.ReportProgress(num_train_steps=total_steps, writer=writer),
    ]

    # Run training loop
    logging.info("Start training loop; Initial compile can take a while...")
    t0 = time.time()
    for step, batch in zip(
        range(initial_step, total_steps + 1),
        input_pipeline.prefetch(ds_train, config.prefetch),
        strict=False,
    ):
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            params_repl, opt_state_repl, loss_repl, update_rng_repl = update_fn_repl(
                params_repl, opt_state_repl, batch, update_rng_repl
            )

        for hook in hooks:
            hook(step)

        if step == initial_step:
            logging.info("First step took %.2f seconds", time.time() - t0)
            t0 = time.time()

        # Report training metrics
        if config.progress_every and step % config.progress_every == 0:
            writer.write_scalars(
                step,
                {
                    "train_loss": float(flax.jax_utils.unreplicate(loss_repl)),
                },
            )
            done = step / total_steps
            logging.info(
                "Step: %d/%d %.1f%%,  ETA: %.2fh",
                step,
                total_steps,
                (100.0 * done),
                ((time.time() - t0) / done * (1.0 - done) / 3600.0),
            )

        # Run Evaluation
        if (config.eval_every and step % config.eval_every == 0) or (
            step == total_steps
        ):
            accuracies = []
            for test_batch in input_pipeline.prefetch(ds_test, config.prefetch):
                logits = infer_fn_repl({"params": params_repl}, test_batch["atoms"])
                accuracies.append(
                    (
                        np.argmax(logits, axis=-1)
                        == np.argmax(test_batch["label"], axis=-1)
                    ).mean()
                )

            accuracy_test = np.mean(accuracies)

            lr = float(lr_fn(step))
            logging.info(
                "Step: %d, Learning rate: %.7e, Test accuracy: %.5f",
                step,
                lr,
                accuracy_test,
            )
            writer.write_scalars(
                step,
                {
                    "accuracy_test": accuracy_test,
                    "lr": lr,
                },
            )

        # Store checkpoint.
        if ((config.checkpoint_every) and (step % config.checkpoint_every == 0)) or (
            step == total_steps
        ):
            ckpt = {
                "params": flax.jax_utils.unreplicate(params_repl),
                "opt_state": flax.jax_utils.unreplicate(opt_state_repl),
                "step": step,
            }

            checkpoint_path = flax_checkpoints.save_checkpoint(workdir, ckpt, step=step)

            logging.info('Store checkpoint at step %d to "%s"', step, checkpoint_path)
    return flax.jax_utils.unreplicate(params_repl)
