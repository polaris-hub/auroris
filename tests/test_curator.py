import os

from alchemy.curation import Curator
from alchemy.curation.actions import Discretization, MoleculeCuration, OutlierDetection
from alchemy.report.broadcaster import HTMLBroadcaster, LoggerBroadcaster


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

    broadcaster = HTMLBroadcaster(report, tmpdir)
    broadcaster.broadcast()
