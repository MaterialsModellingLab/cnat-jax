# Comprehensive Neighbor atom Analysis Transformer

CNAT-JAX is a JAX implementation of the Comprehensive Neighbor atom Analysis
Transformer.

## Installation

```bash
python -m pip install cnat-jax
```

Install the test dependencies with:

```bash
python -m pip install "cnat-jax[test]"
```

## Usage

The command line interface supports training and evaluation:

```bash
cnat-jax --mode=train --workdir=/path/to/workdir --config=/path/to/config.py
cnat-jax --mode=evaluate --workdir=/path/to/workdir
```

## Dataset

CNAT-JAX uses the `na_dataset` TensorFlow Dataset. Follow the dataset
repository for download and preparation instructions:

<https://github.com/MaterialsModellingLab/na_dataset>

After preparing the dataset, set `tfds_data_dir` to the TensorFlow Datasets
directory. If the dataset requires a manual download, set `tfds_manual_dir`
to the directory containing the manually downloaded files. Both values can be
set in the `ml_collections` configuration passed with `--config`.

## Troubleshooting

If a test fails due to a memory error, try:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```
