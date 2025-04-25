import numpy as np
from .tree_node import TreeNode

class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def _mse(self, y):
        """Calculate mean squared error for a set of values"""
        return np.mean((y - np.mean(y)) ** 2)
    
    def _find_best_split(self, X, y):
        """Find the best split for a node"""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                mse = (np.sum((left_y - np.mean(left_y))**2) + 
                      np.sum((right_y - np.mean(right_y))**2))
                
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Build the regression tree recursively"""
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return {'value': np.mean(y)}
            
        feature, threshold = self._find_best_split(X, y)
        if feature is None:
            return {'value': np.mean(y)}
            
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Fit the regression tree"""
        self.tree = self._build_tree(X, y)
        
    def _predict_sample(self, x, node):
        """Predict a single sample"""
        if 'value' in node:
            return node['value']
            
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
            
    def predict(self, X):
        """Predict for multiple samples"""
        return np.array([self._predict_sample(x, self.tree) for x in X])
        
    def _get_terminal_regions(self, X, node, current_region=None):
        """Get terminal regions for samples"""
        if current_region is None:
            current_region = np.ones(len(X), dtype=bool)
            
        if 'value' in node:
            return [current_region]
            
        feature = node['feature']
        threshold = node['threshold']
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_regions = self._get_terminal_regions(X, node['left'], current_region & left_mask)
        right_regions = self._get_terminal_regions(X, node['right'], current_region & right_mask)
        
        return left_regions + right_regions
        
    def get_terminal_regions(self, X):
        """Get terminal regions for samples"""
        return self._get_terminal_regions(X, self.tree) 