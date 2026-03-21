import mlflow
import os
import sys

THRESHOLD = 0.99

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking accuracy for Run ID: {run_id}")

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", None)

if accuracy is None:
    print("No accuracy metric found in MLflow!")
    sys.exit(1)

print(f"Accuracy: {accuracy:.4f} | Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    sys.exit(1)
else:
    print(f"PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}")