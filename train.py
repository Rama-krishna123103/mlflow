import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log the model with MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    print(f"Model saved in run {mlflow.active_run().info.run_id}")
