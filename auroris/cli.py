import datamol as dm
import pandas as pd
import typer

from auroris.curation import Curator
from auroris.report.broadcaster import HTMLBroadcaster

app = typer.Typer()


@app.command()
def curate(config_path: str, dataset_path: str, destination: str, overwrite: bool = False):
    # Load data
    dataset = pd.read_csv(dataset_path)
    curator = Curator.from_json(config_path)

    # Run curation
    dataset, report = curator(dataset)

    # Save dataset
    dm.fs.mkdir(destination, exist_ok=overwrite)
    path = dm.fs.join(destination, "curated.csv")
    dataset.to_csv(path, index=False)

    # Save report as HTML
    report_destination = dm.fs.join(destination, "report")
    broadcaster = HTMLBroadcaster(report, report_destination)
    broadcaster.broadcast()


if __name__ == "__main__":
    app()
