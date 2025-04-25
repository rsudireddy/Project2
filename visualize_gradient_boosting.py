import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from model.gradient_boosting import GradientBoostingClassifier
import seaborn as sns

def plot_learning_curve(X, y):
    """Plot learning curve showing accuracy vs number of trees"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_trees = [10, 20, 50, 100, 200]
    train_scores = []
    test_scores = []
    
    for n in n_trees:
        gb = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, max_depth=3)
        gb.fit(X_train, y_train)
        
        train_pred = gb.predict(X_train)
        test_pred = gb.predict(X_test)
        
        train_scores.append(accuracy_score(y_train, train_pred))
        test_scores.append(accuracy_score(y_test, test_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, train_scores, 'o-', label='Training Accuracy')
    plt.plot(n_trees, test_scores, 'o-', label='Test Accuracy')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.close()

def plot_decision_boundary(X, y):
    """Plot decision boundary for 2D data"""
    if X.shape[1] != 2:
        print("Decision boundary plot requires 2D data")
        return
        
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X, y)
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Predict for each point in the mesh
    Z = gb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.savefig('decision_boundary.png')
    plt.close()

def plot_feature_importance(X, y):
    """Plot feature importance based on tree splits"""
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X, y)
    
    # Count feature usage in splits
    feature_importance = np.zeros(X.shape[1])
    for tree in gb.trees:
        def count_feature_usage(node):
            if 'feature' in node:
                feature_importance[node['feature']] += 1
                count_feature_usage(node['left'])
                count_feature_usage(node['right'])
        count_feature_usage(tree.tree)
    
    # Normalize importance
    feature_importance = feature_importance / feature_importance.sum()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()

def plot_tree_structure(X, y):
    """Plot the structure of the first tree"""
    gb = GradientBoostingClassifier(n_estimators=1, learning_rate=0.1, max_depth=3)
    gb.fit(X, y)
    
    def plot_node(node, x, y, dx, dy):
        if 'value' in node:
            plt.text(x, y, f"Value: {node['value']:.2f}", 
                    bbox=dict(facecolor='white', alpha=0.5))
            return
            
        plt.text(x, y, f"X{node['feature']} <= {node['threshold']:.2f}", 
                bbox=dict(facecolor='white', alpha=0.5))
        
        # Plot left child
        plt.plot([x, x-dx], [y, y-dy], 'k-')
        plot_node(node['left'], x-dx, y-dy, dx/2, dy)
        
        # Plot right child
        plt.plot([x, x+dx], [y, y-dy], 'k-')
        plot_node(node['right'], x+dx, y-dy, dx/2, dy)
    
    plt.figure(figsize=(15, 10))
    plot_node(gb.trees[0].tree, 0, 0, 1, 1)
    plt.axis('off')
    plt.title('First Tree Structure')
    plt.savefig('tree_structure.png')
    plt.close()

def plot_comparison(X, y):
    """Plot comparison between our implementation and scikit-learn's implementation"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_trees = [10, 20, 50, 100, 200]
    our_scores = []
    sklearn_scores = []
    
    for n in n_trees:
        # Our implementation
        our_gb = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, max_depth=3)
        our_gb.fit(X_train, y_train)
        our_pred = our_gb.predict(X_test)
        our_scores.append(accuracy_score(y_test, our_pred))
        
        # scikit-learn implementation
        sklearn_gb = SklearnGBC(n_estimators=n, learning_rate=0.1, max_depth=3, random_state=42)
        sklearn_gb.fit(X_train, y_train)
        sklearn_pred = sklearn_gb.predict(X_test)
        sklearn_scores.append(accuracy_score(y_test, sklearn_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, our_scores, 'o-', label='Our Implementation')
    plt.plot(n_trees, sklearn_scores, 'o-', label='scikit-learn Implementation')
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Accuracy')
    plt.title('Comparison with scikit-learn Implementation')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison.png')
    plt.close()

def plot_training_validation_loss(X, y):
    """Plot training and validation loss vs iterations"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    train_losses = []
    val_losses = []
    
    # Fit the model and track losses
    gb.fit(X_train, y_train)
    for i in range(len(gb.trees)):
        # Calculate training loss
        train_pred = gb.predict_proba(X_train)
        train_loss = -np.mean(y_train * np.log(train_pred[:, 1] + 1e-15) + 
                             (1 - y_train) * np.log(1 - train_pred[:, 1] + 1e-15))
        train_losses.append(train_loss)
        
        # Calculate validation loss
        val_pred = gb.predict_proba(X_val)
        val_loss = -np.mean(y_val * np.log(val_pred[:, 1] + 1e-15) + 
                           (1 - y_val) * np.log(1 - val_pred[:, 1] + 1e-15))
        val_losses.append(val_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Number of Trees')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()

def plot_roc_curve(X, y):
    """Plot ROC curve and calculate AUC"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    
    # Get predicted probabilities
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

def plot_confusion_matrix(X, y):
    """Plot confusion matrix"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_probability_histogram(X, y):
    """Plot histogram of predicted probabilities"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.5, label='Class 0')
    plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.5, label='Class 1')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Histogram of Predicted Probabilities')
    plt.legend()
    plt.grid(True)
    plt.savefig('probability_histogram.png')
    plt.close()

def plot_probability_vs_true(X, y):
    """Plot predicted probabilities vs true labels"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_proba, alpha=0.5)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Probability')
    plt.title('Predicted Probabilities vs True Labels')
    plt.grid(True)
    plt.savefig('probability_vs_true.png')
    plt.close()

def main(n_samples=500):
    # Generate a smaller synthetic dataset
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                             n_redundant=0, n_classes=2, random_state=42)
    
    # Plot learning curve
    plot_learning_curve(X, y)
    
    # Plot decision boundary
    plot_decision_boundary(X, y)
    
    # Plot feature importance
    plot_feature_importance(X, y)
    
    # Plot tree structure
    plot_tree_structure(X, y)
    
    # Plot comparison with scikit-learn
    plot_comparison(X, y)
    
    # Plot training and validation loss
    plot_training_validation_loss(X, y)
    
    # Plot ROC curve
    plot_roc_curve(X, y)
    
    # Plot confusion matrix
    plot_confusion_matrix(X, y)
    
    # Plot probability histogram
    plot_probability_histogram(X, y)
    
    # Plot predicted probabilities vs true labels
    plot_probability_vs_true(X, y)

if __name__ == "__main__":
    main() 