import numpy as np
from .regression_tree import RegressionTree

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.terminal_regions = []
        self.gammas = []
        self.initial_prediction = None
        
    def _compute_initial_prediction(self, y):
        """Initialize f0(x) = argmin_γ sum(L(yi, γ))"""
        # For binary classification with log loss, this is log-odds
        pos = np.mean(y)
        return np.log(pos / (1 - pos))
        
    def _compute_residuals(self, y, predictions):
        """Compute rim = -∂L(yi, f(xi))/∂f(xi)"""
        # For binary classification with log loss, this is y - p
        probas = 1 / (1 + np.exp(-predictions))
        return y - probas
        
    def _compute_optimal_gamma(self, y, predictions, region_mask):
        """Compute γjm = argmin_γ sum(L(yi, fm-1(xi) + γ)) for region Rjm"""
        # For binary classification with log loss, this is:
        # γ = log(sum(y)/sum(1-y)) for the region
        y_region = y[region_mask]
        if len(y_region) == 0:
            return 0
        pos = np.sum(y_region == 1)
        neg = np.sum(y_region == 0)
        if neg == 0:
            return np.log(pos / 1e-10)
        if pos == 0:
            return np.log(1e-10 / neg)
        return np.log(pos / neg)
        
    def fit(self, X, y):
        """Fit the gradient boosting model following Algorithm 10.3 exactly."""
        # Step 1: Initialize f0(x)
        self.initial_prediction = self._compute_initial_prediction(y)
        predictions = np.full(len(y), self.initial_prediction)
        
        # Step 2: For m = 1 to M
        for _ in range(self.n_estimators):
            # Step 2a: Compute residuals
            residuals = self._compute_residuals(y, predictions)
            
            # Step 2b: Fit regression tree to residuals
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Get terminal regions
            terminal_regions = tree.get_terminal_regions(X)
            self.terminal_regions.append(terminal_regions)
            
            # Step 2c: Compute optimal γ for each region
            gammas = []
            for region in terminal_regions:
                gamma = self._compute_optimal_gamma(y, predictions, region)
                gammas.append(gamma)
            self.gammas.append(gammas)
            
            # Step 2d: Update model
            for region, gamma in zip(terminal_regions, gammas):
                predictions[region] += self.learning_rate * gamma
            
            # Store the tree
            self.trees.append(tree)
            
    def predict_proba(self, X):
        """Predict class probabilities."""
        # Start with initial prediction
        predictions = np.full(len(X), self.initial_prediction)
        
        # Add up predictions from all trees
        for tree, terminal_regions, gammas in zip(self.trees, self.terminal_regions, self.gammas):
            # Get terminal regions for new data
            regions = tree.get_terminal_regions(X)
            # Update predictions using optimal gammas
            for region, gamma in zip(regions, gammas):
                predictions[region] += self.learning_rate * gamma
        
        # Convert to probabilities using sigmoid function
        probas = 1 / (1 + np.exp(-predictions))
        return np.vstack([1 - probas, probas]).T
        
    def predict(self, X):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int) 