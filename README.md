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


Auroris is a Python library designed to assist researchers and scientists in managing, cleaning, and preparing data relevant to drug discovery. Auroris will implement a range of techniques to handle, transform, filter, analyze, or visualize the diverse data types commonly encountered in drug discovery. 

Currently, Auroris supports curation for small molecules, with plans to extend to other modalities in drug discovery. The curation module for small molecules includes:

- 🗄️ Molecule Standardization: Ensures that each molecule is represented in a uniform and unambiguous form.

- 🏷️ Detection of Duplicate Molecules with Contradictory Labels: Identifies and resolves inconsistencies in activity data for each molecule.

- ⛰️ Detection of Activity Cliffs Between Stereoisomers: Identifies significant differences in activity between stereoisomers.

- 🔍Outlier Detection and Visualization: Detects and visualizes outliers in molecular activity data.

- 📽️ Visualization of Molecular Distribution in Chemical Space: Provides graphical representations of molecular distributions.

Reproducibility and transparency are core to the mission of Polaris. That’s why with Auroris, you can also automatically generate detailed reports summarizing the changes that happened to a dataset during curation. Through an intuitive API, you can easily define complex curation workflows. Once defined, that workflow is serializable and thus reproducible so you can transparently share how you curated the dataset.
 


## Getting started

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
