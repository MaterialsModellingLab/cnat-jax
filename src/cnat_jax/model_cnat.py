# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array  # Placeholder for jax.Array
PRNGKey = jax.Array
Shape = tuple[int, ...]  # Placeholder for jax.ShapeDtypeStruct
DType = jnp.dtype


class IdentityLayer(nn.Module):
    """Identity layer for residual connections."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    """
    Add position embeddings to the input tensor.


    Attributes:
        posemb_init: Function to initialize the position embeddings.
        param_dtype: Data type of the parameters.
    """

    posemb_init: Callable[[PRNGKey, Shape, DType], Array]
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        # inputs.shape == (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, (
            f"Number of dimensions should be 3, but it is {inputs.ndim}"
        )

        pose_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param(
            "pos_embedding", self.posemb_init, pose_emb_shape, self.param_dtype
        )
        return inputs + pe


class MlpBlock(nn.Module):
    """
    Transformer Multi-Layer Perceptron (MLP) Block / Feedforward Block.

    Attributes:
        mlp_dim: Dimension of the feedforward network.
        dtype: Data type of the parameters.
        param_dtype: Data type of the parameters.
        out_dim: Output dimension of the MLP block.
        dropout_rate: Dropout rate for regularization.
        kernel_init: Function to initialize the kernel weights.
        bias_init: Function to initialize the bias weights.
    """

    mlp_dim: int
    dtype: DType = jnp.float32
    param_dtype: DType = jnp.float32
    out_dim: int | None = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, DType], Array] = (
        nn.initializers.xavier_uniform()
    )
    bias_init: Callable[[PRNGKey, Shape, DType], Array] = nn.initializers.normal(
        stddev=1.0e-6
    )

    @nn.compact
    def __call__(self, inputs, *, deterministic: bool):
        """
        Applies MlpBlock module.

        Args:
            inputs: Input tensor
            deterministic: Whether to apply dropout or not.
        """
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)

        return output


class Encoder1DBlock(nn.Module):
    """
    Transformer Encoder Block.

    Attributes:
        mlp_dim: Dimension of the feedforward network.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for regularization.
        add_position_embedding: Whether to add position embedding or not.
    """

    mlp_dim: int
    num_heads: int
    dtype: DType = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, inputs, *, mask: Array | None = None, deterministic: bool | None = None
    ):
        """
        Applies Encoder1DBlock module.

        Args:
            x: Input tensor of shape (batch, seq, hidden).
            mask: Optional mask for attention.
            deterministic: Whether to apply dropout or not.
        """
        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
        )(x, x, mask=mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
        )(y, deterministic=deterministic)

        return x + y


class Encoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.

    Attributes:
        num_layers: Number of layers in the encoder.
        mlp_dim: Dimension of the feedforward network.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for regularization.
        attention_dropout_rate: Dropout rate for attention.
        add_position_embedding: Whether to add position embedding or not.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = False

    @nn.compact
    def __call__(self, inputs, *, mask: Array | None = None, train: bool | None = None):
        """
        Applies Encoder module.
        Args:
            inputs: Input tensor of shape (batch, seq, emb_dim).
            train: Set to True for training mode.
        Returns:
            x: Output tensor of shape (batch, seq, emb_dim).
        """
        assert inputs.ndim == 3, f"Expected (batch, seq, emb) got {inputs.shape}"
        x = inputs

        if self.add_position_embedding:
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02), name="posembed_input"
            )(x)

            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoder_block_{lyr}",
                num_heads=self.num_heads,
            )(x, mask=mask, deterministic=not train)

        encoded = nn.LayerNorm(name="encoder_norm")(x)
        return encoded


class CnaTransformer(nn.Module):
    """CnaTransformer model."""

    num_classes: int
    transformer: Any
    hidden_size: int
    resnet: Any = None
    representation_size: int | None = None
    classifier: str = "token"
    head_bias_init: float = 0.0
    encoder: type[nn.Module] = Encoder
    model_name: str | None = None

    @nn.compact
    def __call__(self, inputs, *, train):
        x = inputs

        x = nn.Dense(features=self.hidden_size, name="embedding")(x)

        if self.transformer is not None:
            n, _, p = x.shape

            if self.classifier in ["token", "token_unpooled"]:
                cls = self.param("cls", nn.initializers.zeros, (1, 1, p))
                cls = jnp.tile(cls, (n, 1, 1))
                x = jnp.concatenate([cls, x], axis=1)

            x = self.encoder(name="Transformer", **self.transformer)(x, train=train)

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))
        elif self.classifier in ["unpooled", "token_unpooled"]:
            pass
        else:
            raise ValueError(f"Invalid classifier type: {self.classifier}")

        if self.representation_size is not None:
            x = nn.Dense(features=self.representation_size, name="pre_logits")(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name="pre_logits")(x)

        if self.num_classes:
            x = nn.Dense(
                features=self.num_classes,
                name="head",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(self.head_bias_init),
            )(x)
        return x
