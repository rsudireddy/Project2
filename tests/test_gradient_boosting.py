import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_data(filename):
    """Load test data from CSV file"""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), filename))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def test_toy_data():
    """Test with small toy dataset"""
    X, y = load_data("small_data.csv")
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=1)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy >= 0.75, f"Accuracy {accuracy:.2f} is below threshold 0.75"

def test_noisy_data():
    """Test with noisy data"""
    X, y = load_data("noisy_data.csv")
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=2)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy >= 0.70, f"Accuracy {accuracy:.2f} is below threshold 0.70"

def test_collinear_data():
    """Test with highly collinear features"""
    X, y = load_data("collinear_data.csv")
    gb = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=2)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy >= 0.75, f"Accuracy {accuracy:.2f} is below threshold 0.75"

def test_imbalanced_data():
    """Test with imbalanced classes"""
    X, y = load_data("imbalance_data.csv")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    f1 = f1_score(y, y_pred)
    assert f1 >= 0.3, f"F1 score {f1:.2f} is below threshold 0.3"

def test_nonlinear_data():
    """Test with non-linear decision boundary"""
    X, y = load_data("nonlinear_data.csv")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy >= 0.70, f"Accuracy {accuracy:.2f} is below threshold 0.70"

def test_edge_cases():
    """Test edge cases"""
    # Single feature
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    gb = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=1)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    assert accuracy_score(y, y_pred) == 1.0, "Should perfectly fit single feature data"
    
    # Extreme regularization
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    gb = GradientBoostingClassifier(n_estimators=5, learning_rate=0.01, max_depth=1)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    assert accuracy_score(y, y_pred) >= 0.5, "Should at least do better than random"

def test_probability_outputs():
    """Test probability outputs"""
    X, y = load_data("sample_data.csv")
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2)
    gb.fit(X, y)
    proba = gb.predict_proba(X)
    
    # Check probability bounds
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities should be between 0 and 1"
    
    # Check that probabilities sum to 1
    assert np.allclose(np.sum(proba, axis=1), 1), "Probabilities should sum to 1"
    
    # Check probability consistency with predictions
    y_pred = gb.predict(X)
    assert np.all((proba[:, 1] >= 0.5) == y_pred), "Predictions should match probabilities >= 0.5"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 