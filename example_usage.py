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
