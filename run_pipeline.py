"""
PURPOSE:
    Command-line entry point that triggers the full ZenML training pipeline.
    After the pipeline finishes, it prints the MLflow tracking URI so you can
    inspect experiment runs in the MLflow UI.
"""

import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
def main():
    # Run the ML pipeline and start the MLflow UI for experiment tracking.

    run = ml_pipeline()
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )

if __name__ == "__main__":
    main()