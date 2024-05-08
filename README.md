# Auroris

[![PyPI](https://img.shields.io/pypi/v/auroris)](https://pypi.org/project/auroris/)
[![Conda](https://img.shields.io/conda/v/conda-forge/auroris?label=conda&color=success)](https://anaconda.org/conda-forge/auroris)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/auroris)](https://pypi.org/project/auroris/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/auroris)](https://anaconda.org/conda-forge/auroris)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/auroris)](https://pypi.org/project/auroris/)

[![test](https://github.com/polaris-hub/polaris/actions/workflows/test.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/test.yml)
[![release](https://github.com/polaris-hub/polaris/actions/workflows/release.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/release.yml)
[![code-check](https://github.com/polaris-hub/polaris/actions/workflows/code-check.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/code-check.yml)
[![doc](https://github.com/polaris-hub/polaris/actions/workflows/doc.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/doc.yml)

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