import mlflow
import mlflow.sklearn
import numpy as np

# Load the model using its run ID
run_id = "your_run_id_here"  # Replace with the actual run ID from MLflow
model_uri = f"runs:/{run_id}/model"

# Load the model
model = mlflow.sklearn.load_model(model_uri)

# Prepare test data (or use real test data)
X_test = np.array([[5.1, 3.5, 1.4, 0.2],  # Example data from the Iris dataset
                   [6.2, 2.9, 4.3, 1.3]])

# Make predictions
predictions = model.predict(X_test)

# Print the predictions
print("Predictions:", predictions)
