# Gradient Boosting Classifier Implementation

This project implements a Gradient Boosting Classifier from scratch in Python, along with comprehensive testing and visualization tools. The implementation includes various features like regression trees, gradient boosting, and model evaluation metrics.

## Features

- Custom implementation of Gradient Boosting Classifier
- Regression Tree implementation
- Comprehensive test suite
- Data generation utilities
- Visualization tools for:
  - Learning curves
  - Decision boundaries
  - Feature importance
  - Tree structure
  - Model comparison with scikit-learn
  - Training/validation loss
  - ROC curves
  - Confusion matrices
  - Probability distributions

## Requirements

- Python 3.8+
- Required packages (see `requirements.txt`):
  - numpy
  - scikit-learn
  - pytest
  - matplotlib
  - pandas
  - seaborn

## Installation


cd Project2_ML

1. Create and activate a virtual environment:

python3 -m venv venv
venv\Scripts\activate

2. Install dependencies:

pip install -r requirements.txt


## Project Structure

```
Project2_ML/
├── model/
│   ├── gradient_boosting.py    # Main gradient boosting implementation
│   ├── regression_tree.py      # Regression tree implementation
│   └── tree_node.py            # Tree node structure
├── tests/
│   ├── test_gradient_boosting.py  # Test cases
│   └── various test datasets
├── generate_classification_data.py  # Data generation script
├── example_usage.py            # Example usage script
├── visualize_gradient_boosting.py  # Visualization script
├── run_project.py              # Script to run entire project
└── requirements.txt            # Project dependencies
```

## Usage

### 1. Running the Entire Project

To run all components of the project (data generation, tests, example usage, and visualizations):

python3 run_project.py


This will:
- Generate test datasets
- Run all tests
- Execute the example usage script
- Generate all visualizations

### 2. Generating Test Data

To generate test datasets:

python3 generate_classification_data.py

This creates several test datasets in the `/tests` directory:
- `small_data.csv`: Simple toy dataset
- `noisy_data.csv`: Dataset with added noise
- `collinear_data.csv`: Dataset with collinear features
- `imbalance_data.csv`: Imbalanced class distribution
- `nonlinear_data.csv`: Non-linear decision boundary
- `sample_data.csv`: General sample dataset

### 3. Running Tests

To run the test suite:

python3 -m pytest tests/test_gradient_boosting.py -v

Tests cover:
- Toy data
- Noisy data
- Collinear features
- Imbalanced classes
- Non-linear boundaries
- Edge cases
- Probability outputs

### 4. Example Usage


python3 example_usage.py


This demonstrates:
- Model initialization
- Training
- Prediction
- Evaluation metrics

### 5. Generating Visualizations

To generate all visualizations:

python3 visualize_gradient_boosting.py


This creates several visualization files:
- `learning_curve.png`: Accuracy vs number of trees
- `decision_boundary.png`: 2D decision boundary
- `feature_importance.png`: Feature importance plot
- `tree_structure.png`: First tree structure
- `comparison.png`: Comparison with scikit-learn
- `loss_curves.png`: Training/validation loss
- `roc_curve.png`: ROC curve with AUC
- `confusion_matrix.png`: Confusion matrix
- `probability_histogram.png`: Probability distribution
- `probability_vs_true.png`: Probabilities vs true labels

## Customizing Parameters

You can modify various parameters in the scripts:

1. In `example_usage.py`:
```python
gb = GradientBoostingClassifier(
    n_estimators=100,    # Number of trees
    learning_rate=0.1,   # Learning rate
    max_depth=3         # Maximum tree depth
)
```

2. In `visualize_gradient_boosting.py`:
```python
def main(n_samples=500):  # Number of samples for visualization
    # ...
```

3. In `generate_classification_data.py`:
```python
def generate_all_datasets(n_samples=100):  # Number of samples for each dataset
    # ...
```

## Project Questions and Answers

### 1. What does the model you have implemented do and when should it be used?

**Answer:**
We've implemented a binary Gradient-Boosting Tree classifier that:
- Starts with a constant log-odds prediction
- Iteratively fits simple regressors (depth-1 "stumps") to the residuals of the logistic loss
- Combines predictions through a sigmoid function to yield probabilities
- Uses a 0.5 cutoff for binary classification

This model is particularly useful when:
- You need to capture both linear and non-linear decision boundaries
- You want implicit feature selection (especially with stumps)
- You need robustness to outliers and heterogeneous feature scales
- You're working with structured/tabular data
- You require high predictive accuracy

### 2. How did you test your model to determine if it is working reasonably correctly?

**Answer:**
We implemented seven comprehensive test scenarios:

1. **Toy Data Test**
   - Basic sanity check
   - Minimum accuracy requirement: 75%

2. **Noisy Data Test**
   - Tests noise robustness
   - Minimum accuracy requirement: 70%

3. **Collinear Features Test**
   - Evaluates feature selection behavior
   - Minimum accuracy requirement: 75%

4. **Imbalanced Classes Test**
   - Tests handling of 90/10 class distribution
   - Minimum accuracy requirement: 65%

5. **Non-linear Data Test**
   - Tests complex boundary learning
   - Minimum accuracy requirement: 70%

6. **Scikit-learn Comparison**
   - Validates against a library implementation
   - Ensures predictions and metrics align within tolerance

7. **Edge Cases Test**
   - Tests single-feature inputs
   - Tests extreme regularization
   - Verifies coefficient sparsity

Each test includes visual verification through "True vs. Predicted" plots to ensure no invalid outputs are produced.

### 3. What parameters have you exposed to users of your implementation in order to tune performance?

**Answer:**
The model exposes several key parameters for tuning:

1. **Core Parameters:**
   - `n_estimators`: Number of boosting iterations
   - `learning_rate`: Shrinkage factor for each new learner
   - `max_depth`: Maximum depth of each tree

2. **Example Usage:**
```python
from model.gradient_boosting import GradientBoostingClassifier

# Initialize the model
gb = GradientBoostingClassifier(
    n_estimators=100,    # Number of trees
    learning_rate=0.1,   # Learning rate
    max_depth=3         # Maximum tree depth
)

# Fit the model
gb.fit(X_train, y_train)

# Make predictions
predictions = gb.predict(X_test)
probabilities = gb.predict_proba(X_test)
```

### 4. Are there specific inputs that your implementation has trouble with?

**Answer:**
Yes, there are some challenging scenarios:

1. **Severe Class Imbalance**
   - Current Issue: May predict only majority class under default 0.5 cutoff
   - Potential Solution: Implement class weights or resampling techniques

2. **High-Dimensional Sparse Inputs**
   - Current Issue: Stump-based splits may be uninformative
   - Potential Solution: Add feature importance-based selection

3. **Noisy/Overlapping Classes**
   - Current Issue: Risk of overfitting
   - Potential Solution: Implement early stopping with validation set

4. **Missing Values**
   - Current Issue: No native handling
   - Potential Solution: Add imputation strategies

5. **Categorical Features**
   - Current Issue: Requires one-hot encoding
   - Potential Solution: Add native categorical feature support

Given more time, we would prioritize:
1. Implementing early stopping
2. Adding class weights for imbalance
3. Developing feature importance metrics
4. Adding native categorical feature support

## Team Members
- Sudireddy Raghavender Reddy (A20554654)
- Chaitanya Durgesh Nynavarapu (A20561894)
- Purnachandra Reddy Peddasura (A20544751)
- Jeswanth Jayavarapu (A20547505)
