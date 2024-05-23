import os

from pandas.core.api import DataFrame as DataFrame

from auroris.curation import Curator
from auroris.curation.actions import Discretization, MoleculeCuration, OutlierDetection
from auroris.report.broadcaster import HTMLBroadcaster, LoggerBroadcaster

try:
    import jinja2
except ImportError:
    jinja2 = None


def test_curator_save_load(tmpdir):
    curator = Curator(
        steps=[
            MoleculeCuration(input_column="smiles"),
            OutlierDetection(method="zscore", columns=["outlier_column"]),
        ],
    )
    path = os.path.join(tmpdir, "curator.json")
    curator.to_json(path)
    curator_reload = curator.from_json(path)

    assert len(curator.steps) == len(curator_reload.steps)
    for step1, step2 in zip(curator.steps, curator_reload.steps):
        assert step1 == step2

    assert curator.steps[1].method == "zscore"
    assert curator.steps[1].columns == ["outlier_column"]


def test_curator_integration(dataset, tmpdir):
    curator = Curator(
        steps=[
            OutlierDetection(method="zscore", columns=["outlier_column"]),
            MoleculeCuration(input_column="smiles"),
            Discretization(input_column="outlier_column", thresholds=[0.0]),
        ],
    )
    dataset, report = curator.transform(dataset)
    assert len(report.sections) == 3

    broadcaster = LoggerBroadcaster(report)
    broadcaster.broadcast()

    if jinja2:
        broadcaster = HTMLBroadcaster(report, tmpdir)
        broadcaster.broadcast()
