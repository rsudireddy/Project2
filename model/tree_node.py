import numpy as np

class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.value = value             # Prediction value for leaf nodes
        self.left = left               # Left child node
        self.right = right             # Right child node
        
    def is_leaf(self):
        return self.value is not None 