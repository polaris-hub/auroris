# Auroris

[![PyPI](https://img.shields.io/pypi/v/auroris)](https://pypi.org/project/auroris/)
[![Conda](https://img.shields.io/conda/v/conda-forge/auroris?label=conda&color=success)](https://anaconda.org/conda-forge/auroris)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/auroris)](https://pypi.org/project/auroris/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/auroris)](https://anaconda.org/conda-forge/auroris)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/auroris)](https://pypi.org/project/auroris/)

[![test](https://github.com/polaris-hub/auroris/actions/workflows/test.yml/badge.svg)](https://github.com/polaris-hub/auroris/actions/workflows/test.yml)
[![release](https://github.com/polaris-hub/auroris/actions/workflows/release.yml/badge.svg)](https://github.com/polaris-hub/auroris/actions/workflows/release.yml)
[![code-check](https://github.com/polaris-hub/auroris/actions/workflows/code-check.yml/badge.svg)](https://github.com/polaris-hub/auroris/actions/workflows/code-check.yml)
[![doc](https://github.com/polaris-hub/auroris/actions/workflows/doc.yml/badge.svg)](https://github.com/polaris-hub/auroris/actions/workflows/doc.yml)

Tools for data curation in the Polaris ecosystem. 


### Getting started

```python
from auroris.curation import Curator
from auroris.curation.actions import MoleculeCuration, OutlierDetection, Discretization

# Define the curation workflow
curator = Curator(
    steps=[
        MoleculeCuration(input_column="smiles"),
        OutlierDetection(method="zscore", columns=["SOL"]),
        Discretization(input_column="SOL", thresholds=[-3]),
    ],
    parallelized_kwargs = { "n_jobs": -1 }
)

# Run the curation
dataset, report = curator(dataset)
```
### Run curation with command line
A `Curator` object is serializable, so you can save it to and load it from a JSON file to reproduce the curation.

```
auroris [config_file] [destination] --dataset-path [data_path]
```

## Documentation

Please refer to the [documentation](https://polaris-hub.github.io/auroris/), which contains tutorials for getting started with `auroris` and detailed descriptions of the functions provided.

## Installation

You can install `auroris` using conda/mamba/micromamba:

```bash
conda install -c conda-forge auroris
```

You can also use pip:

```bash
pip install auroris
```

## Development lifecycle

### Setup dev environment

```shell
conda env create -n auroris -f env.yml
conda activate auroris

pip install --no-deps -e .
```

<details>
  <summary>Other installation options</summary>
  
    Alternatively, using [uv](https://github.com/astral-sh/uv):
    ```shell
    uv venv -p 3.12 auroris
    source .venv/auroris/bin/activate
    uv pip compile pyproject.toml -o requirements.txt --all-extras
    uv pip install -r requirements.txt 
    ```   
</details>


### Tests

You can run tests locally with:

```shell
pytest
```

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).
