from typing import Optional

import datamol as dm
import typer

from auroris.curation import Curator
from auroris.report.broadcaster import HTMLBroadcaster

app = typer.Typer()


@app.command()
def curate(config_path: str, destination: str, dataset_path: Optional[str] = None, overwrite: bool = False):
    # Create the curator
    curator = Curator.from_json(config_path)

    # Overwrite the source dataset if it is set
    if dataset_path is not None:
        curator.src_dataset_path = dataset_path

    # Run curation
    dataset, report = curator.transform()

    # Save dataset
    dm.fs.mkdir(destination, exist_ok=overwrite)
    path = dm.fs.join(destination, "curated.parquet")
    dataset.to_parquet(path, index=False)

    # Save a copy of the curation config
    config_destination = dm.fs.join(destination, "config.json")
    curator.to_json(config_destination)

    # Save report as HTML
    report_destination = dm.fs.join(destination, "report")
    broadcaster = HTMLBroadcaster(report, report_destination)
    broadcaster.broadcast()


if __name__ == "__main__":
    app()
