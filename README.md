# Project 2

## Team Members
Sudireddy Raghavender Reddy (A20554654) 

Chaitanya Durgesh Nynavarapu(A20561894)

Purnachandra Reddy Peddasura(A20544751)

jeswanth jayavarapu (A20547505)


## Steps to run project
1) python -m venv venv
2) venv\Scripts\activate
3) pip install -r requirements.txt
4) python generate_classification_data.py
5) cd tests
6) pytest



## For Example usage
cd ..
python example_usage.py

## For Analysing the model
python Analysis.py

Answer the following questions.

* What does the model you have implemented do and when should it be used?

We’ve built a binary Gradient-Boosting Tree classifier. Internally, it begins with a constant log-odds prediction and then iteratively fits simple regressors (by default depth-1 “stumps”) to the residuals of the logistic loss. The ensemble’s summed output is passed through a sigmoid to yield probabilities, and we apply a 0.5 cutoff for hard 0/1 labels.

We would use this model when we need a flexible, high-accuracy classifier that can:

Capture both linear and non-linear decision boundaries
Perform implicit feature selection (especially with stumps)
Be robust to outliers and heterogeneous feature scales
Serve as a learning exercise in understanding the mechanics of boosting

* How did you test your model to determine if it is working reasonably correctly?
We wrote seven PyTest scenarios, each with its own dataset and minimum‐accuracy threshold:

i) Toy data: basic sanity check, ≥75% accuracy

ii) Synthetic random & noisy data: noise robustness, ≥70% accuracy

iii) Highly collinear features: feature-selection behavior, ≥75% accuracy

iv) Imbalanced classes (90/10): skew handling, ≥65% accuracy

v) Non-linear (radial) data: complex boundary, ≥70% accuracy

vi) Scikit-learn comparison: small random set comparison to a library implementation (predictions and metrics align within tolerance)

vii) Edge cases: single-feature inputs, extreme regularization, coefficient sparsity checks

For each test, we also generated a “True vs. Predicted” plot (with y-axis clipped at zero) so we could visually verify that no invalid (negative) outputs were produced.

* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
n_estimators/n_rounds: number of boosting iterations
learning_rate/lr: shrinkage factor on each new learner
base_learner_cls & base_learner_params: allows swapping in deeper trees (e.g., depth-2/3) or any custom regressor.

example usage:
import numpy as np
from model.GradientBoostingTree import BoostingClassifier
features = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
labels = np.array([0, 0, 1, 1])
classifier = BoostingClassifier(n_rounds=20, lr=0.1)
classifier.fit(features, labels)
predictions = classifier.predict(features)
probabilities = classifier.predict_proba(features)
print("Predicted labels:      ", predictions)
print("Predicted probabilities:", probabilities)


* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?


Severe class imbalance can cause the model to predict only the majority class under the default 0.5 cutoff, though we mitigate this by scanning for the F1-optimal threshold. Very high-dimensional, sparse inputs make stump-based splits uninformative when most features are zero. Extremely noisy or heavily overlapping classes risk overfitting unless we limit boosting rounds or add regularization.

Given more time, we’d add early stopping on a held-out validation set to halt training once performance plateaus.
We’d incorporate class weights or resampling techniques to handle extreme imbalance.


