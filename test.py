import mlflow
import mlflow.sklearn
import numpy as np

# Search for the most recent run
runs = mlflow.search_runs(order_by=["start_time desc"])
latest_run = runs.iloc[0]  # Get the most recent run
run_id = latest_run.run_id

# Construct the model URI and load the model
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully!")
# Prepare test data (or use real test data)
X_test = np.array([[5.1, 3.5, 1.4, 0.2],  # Example data from the Iris dataset
                   [6.2, 2.9, 4.3, 1.3]])

# Make predictions
predictions = model.predict(X_test)

# Print the predictions
print("Predictions:", predictions)
