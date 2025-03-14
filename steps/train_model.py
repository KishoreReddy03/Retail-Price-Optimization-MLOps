from typing import List, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.logger import get_logger
from materializers.custom_materializer import ListMaterializer, SKLearnModelMaterializer


experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

@step(experiment_tracker="mlflow_tracker_mlops",
  settings={"experiment_tracker.mlflow": {"experiment_name": "test_name"}},
  enable_cache=False, output_materializers=[SKLearnModelMaterializer, ListMaterializer])
def sklearn_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"]
) -> Tuple[
    Annotated[LinearRegression, "model"],
    Annotated[List[str], "predictors"],
]:
    """Trains a linear regression model and outputs the summary."""
    try:
        mlflow.end_run()  # End any existing run
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()  # Automatically logs all sklearn parameters, metrics, and models
            model = LinearRegression()
            model.fit(X_train, y_train)  # train the model
            # Note: You might need to modify the predictors logic as per sklearn model
            predictors = X_train.columns.tolist()  # considering all columns in X_train as predictors 
            print(predictors)
            print(model.predict(X_train))
            return model, predictors
    except Exception as e:
        logger.error(e)
        raise e
