import os

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
            OutlierDetection(method="zscore", columns=["outlier_column"]),
            MoleculeCuration(input_column="smiles"),
        ],
    )
    path = os.path.join(tmpdir, "curator.json")
    curator.to_json(path)
    curator.from_json(path)

    assert len(curator.steps) == 2
    assert curator.steps[0].method == "zscore"
    assert curator.steps[0].columns == ["outlier_column"]


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
