import os

from alchemy.curation import Curator
from alchemy.curation.actions import MoleculeCuration, OutlierDetection
from alchemy.report.broadcaster import LoguruBroadcaster


def test_curator_save_load(tmpdir):
    curator = Curator(
        actions=[
            OutlierDetection(method="zscore", columns=["outlier_column"]),
        ]
    )
    path = os.path.join(tmpdir, "curator.json")
    curator.to_json(path)
    curator.from_json(path)

    assert len(curator.actions) == 1
    assert curator.actions[0].method == "zscore"
    assert curator.actions[0].columns == ["outlier_column"]


def test_curator_integration(dataset):
    curator = Curator(
        actions=[
            OutlierDetection(method="zscore", columns=["outlier_column"]),
            MoleculeCuration(input_column="smiles"),
        ]
    )
    dataset, report = curator.run(dataset)
    assert len(report.sections) == 2

    broadcaster = LoguruBroadcaster()
    broadcaster.broadcast(report)
