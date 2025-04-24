import pandas as pd
import numpy as np
import os


def generate_sample_data(n=100, features=3, seed=123):
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, (n, features))
    weights = np.array([1.5, -2.0, 1.0])
    noise = np.random.normal(0, 0.2, size=n)
    y = (X @ weights[:features] + noise > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(features)])
    df['y'] = y
    return df


def generate_small_data():
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5]])
    y = np.array([0, 0, 1, 1])
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df['y'] = y
    return df


def generate_noisy_data(n, features, seed):
    np.random.seed(seed)
    X = np.random.randn(n, features)
    true_weights = np.random.randn(features)
    linear_comb = X @ true_weights
    probs = 1 / (1 + np.exp(-linear_comb))
    y = (probs > 0.5).astype(int)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(features)])
    df['y'] = y
    return df


def generate_collinear_data(n, seed):
    np.random.seed(seed)
    X1 = np.random.rand(n)
    X2 = X1 + np.random.normal(0, 0.01, n)
    X3 = np.random.rand(n)
    y = (X1 + X3 > 1).astype(int)
    df = pd.DataFrame({'x0': X1, 'x1': X2, 'x2': X3, 'y': y})
    return df


def generate_imbalance_data(n, features, seed):
    np.random.seed(seed)
    X = np.random.randn(n, features)
    y = np.concatenate((np.zeros(int(n * 0.9)), np.ones(int(n * 0.1))))
    np.random.shuffle(y)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(features)])
    df['y'] = y
    return df


def generate_nonlinear_data(n, seed):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, (n, 2))
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = (radius > 0.5).astype(int)
    df = pd.DataFrame(X, columns=['x0', 'x1'])
    df['y'] = y
    return df


def save_dataset(df, filename):
    os.makedirs('tests', exist_ok=True)
    df.to_csv(f'tests/{filename}', index=False)
    print(f"{filename} generated!")


def main():
    print("Generating all datasets...")

    save_dataset(generate_small_data(), 'small_data.csv')
    save_dataset(generate_noisy_data(100, 3, 42), 'noisy_data.csv')
    save_dataset(generate_collinear_data(100, 42), 'collinear_data.csv')
    save_dataset(generate_imbalance_data(200, 3, 42), 'imbalance_data.csv')
    save_dataset(generate_nonlinear_data(200, 42), 'nonlinear_data.csv')
    save_dataset(generate_sample_data(), 'sample_data.csv')

    print("All datasets generated in /tests folder.")


if __name__ == "__main__":
    main()
