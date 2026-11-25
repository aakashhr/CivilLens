"""Train the local toxicity classifier and save it to models/toxicity_model.joblib.

Usage: python train.py
"""
from civillens.toxicity import train_model

if __name__ == "__main__":
    _, m = train_model()
    print("Saved models/toxicity_model.joblib and reports/toxicity_metrics.json")
