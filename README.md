# Auroris

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