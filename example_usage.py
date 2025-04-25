import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model.gradient_boosting import GradientBoostingClassifier

def main(n_samples=1000):
    # Generate synthetic data
    X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2,
                             random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       random_state=42)
    
    # Initialize the model
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                  max_depth=3)
    
    # Fit the model
    print("Fitting the model...")
    gb.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nFirst 5 predictions and probabilities:")
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  True label: {y_test[i]}")
        print(f"  Predicted label: {y_pred[i]}")
        print(f"  Probability of class 1: {y_proba[i, 1]:.4f}")

if __name__ == "__main__":
    main()
