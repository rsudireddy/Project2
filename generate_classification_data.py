import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification


def generate_sample_data(n_samples=100, features=3, seed=123):
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, (n_samples, features))
    weights = np.array([1.5, -2.0, 1.0])
    noise = np.random.normal(0, 0.2, size=n_samples)
    y = (X @ weights[:features] + noise > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(features)])
    df['y'] = y
    return df


def generate_small_data(n_samples=100):
    """Generate a small toy dataset"""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    df = pd.DataFrame(X, columns=['X'])
    df['y'] = y
    df.to_csv('tests/small_data.csv', index=False)
    print("small_data.csv generated!")


def generate_noisy_data(n_samples=100):
    """Generate dataset with noise"""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                             n_redundant=0, n_classes=2, random_state=42,
                             class_sep=1.0)  # Reduce class separation for more noise
    # Add random noise to features
    X += np.random.normal(0, 0.5, X.shape)
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['y'] = y
    df.to_csv('tests/noisy_data.csv', index=False)
    print("noisy_data.csv generated!")


def generate_collinear_data(n_samples=100):
    """Generate dataset with collinear features"""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                             n_redundant=0, n_classes=2, random_state=42)
    # Make features collinear
    X[:, 1] = X[:, 0] + np.random.normal(0, 0.1, size=n_samples)
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['y'] = y
    df.to_csv('tests/collinear_data.csv', index=False)
    print("collinear_data.csv generated!")


def generate_imbalance_data(n_samples=100):
    """Generate imbalanced dataset"""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                             n_redundant=0, n_classes=2, weights=[0.9, 0.1],
                             random_state=42)
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['y'] = y
    df.to_csv('tests/imbalance_data.csv', index=False)
    print("imbalance_data.csv generated!")


def generate_nonlinear_data(n_samples=100):
    """Generate dataset with nonlinear decision boundary"""
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2) > 1).astype(int)
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['y'] = y
    df.to_csv('tests/nonlinear_data.csv', index=False)
    print("nonlinear_data.csv generated!")


def generate_sample_data(n_samples=100):
    """Generate a sample dataset"""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                             n_redundant=0, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['y'] = y
    df.to_csv('tests/sample_data.csv', index=False)
    print("sample_data.csv generated!")


def generate_all_datasets(n_samples=100):
    """Generate all datasets"""
    print("Generating all datasets...")
    generate_small_data(n_samples)
    generate_noisy_data(n_samples)
    generate_collinear_data(n_samples)
    generate_imbalance_data(n_samples)
    generate_nonlinear_data(n_samples)
    generate_sample_data(n_samples)
    print("All datasets generated in /tests folder.")


if __name__ == "__main__":
    generate_all_datasets()
