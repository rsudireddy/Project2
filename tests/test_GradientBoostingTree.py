import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make sure we can import from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.GradientBoostingTree import BoostingClassifier

# Directory for saving plots
FIG_DIR = "plots"
os.makedirs(FIG_DIR, exist_ok=True)


def execute_test(data_file, rounds=50, lr=0.1, required_accuracy=0.7, figure_name=None):
    """Run a full test on one CSV file."""
    df = pd.read_csv(data_file)
    X = df.drop("y", axis=1).values
    y = df["y"].values

    # Train the model
    booster = BoostingClassifier(n_rounds=rounds, lr=lr)
    booster.fit(X, y)

    # Make and clip predictions
    raw_preds = booster.predict(X)
    preds     = np.clip(raw_preds, 0, 1)

    # Compute accuracy
    acc = np.mean(preds == y)
    print(f"Accuracy on {data_file}: {acc:.4f}")

    # Sanity checks
    assert preds.shape == y.shape
    assert acc >= required_accuracy, f"Acc {acc:.4f} < {required_accuracy}"
    assert np.all(preds >= 0), f"Negative preds in {data_file}!"

    # Optional plot
    if figure_name:
        plt.figure()
        plt.scatter(range(len(y)), y,    c='blue', label='True')
        plt.scatter(range(len(preds)), preds, c='red', marker='x', label='Pred')
        plt.title(f"{data_file} Results")
        plt.ylim(bottom=0)
        plt.legend()
        plt.savefig(os.path.join(FIG_DIR, figure_name))
        plt.close()


def test_toy_data():
    """Manual toy dataset test."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    booster = BoostingClassifier(n_rounds=10, lr=0.5)
    booster.fit(X, y)

    p = np.clip(booster.predict(X), 0, 1)
    prob = np.clip(booster.predict_proba(X), 0, 1)

    assert p.shape == y.shape
    assert np.sum(p == y) >= 3
    assert np.all(p >= 0)

    plt.figure()
    plt.scatter(X[:, 0], y,   c='blue', label='True')
    plt.scatter(X[:, 0], prob,c='red',  label='Prob')
    plt.title("Toy Data")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "toy_data.png"))
    plt.close()


def test_sample_data():
    execute_test("sample_data.csv",   rounds=50,  lr=0.1, required_accuracy=0.7,  figure_name="sample.png")

def test_simple_data():
    execute_test("small_data.csv",    rounds=10,  lr=0.5, required_accuracy=1.0,  figure_name="simple.png")

def test_noisy_data():
    execute_test("noisy_data.csv",    rounds=100, lr=0.1, required_accuracy=0.7,  figure_name="noisy.png")

def test_collinear_data():
    execute_test("collinear_data.csv",rounds=50,  lr=0.1, required_accuracy=0.75, figure_name="collinear.png")

def test_imbalance_data():
    execute_test("imbalance_data.csv",rounds=100, lr=0.1, required_accuracy=0.65, figure_name="imbalance.png")

def test_nonlinear_data():
    execute_test("nonlinear_data.csv",rounds=150, lr=0.1, required_accuracy=0.7,  figure_name="nonlinear.png")
