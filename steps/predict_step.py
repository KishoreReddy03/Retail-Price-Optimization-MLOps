import numpy as np
import pandas as pd
from zenml.steps import step
from zenml.integrations.bentoml.model_deployers import BentoMLModelDeployer
from rich import print as rich_print

@step
def predictor(
    inference_data: pd.DataFrame,
    deployer: BentoMLModelDeployer,
) -> np.ndarray:
    """Run an inference request against the BentoML prediction service.

    Args:
        deployer: The BentoML model deployer.
        inference_data: The data to predict.
    """
    # Get the active deployment service
    services = deployer.find_model_server()
    if not services:
        raise RuntimeError("No active BentoML service found!")
    
    service = services[0]  # Use the first available service
    
    service.start(timeout=10)  # Ensures service is running
    
    # Convert inference data to numpy array
    inference_data = inference_data.to_numpy()
    
    # Make a prediction using the service
    prediction = service.predict("predict_ndarray", inference_data)
    
    rich_print("Prediction: ", prediction)
    return prediction
