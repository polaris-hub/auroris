import os

from alchemy.curation import Curator
from alchemy.curation.actions import MoleculeCuration, OutlierDetection
from alchemy.report.broadcaster import LoggerBroadcaster


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


def test_curator_integration(dataset):
    curator = Curator(
        steps=[
            OutlierDetection(method="zscore", columns=["outlier_column"]),
            MoleculeCuration(input_column="smiles"),
        ],
    )
    dataset, report = curator.transform(dataset)
    assert len(report.sections) == 2

    broadcaster = LoggerBroadcaster(report)
    broadcaster.broadcast()
