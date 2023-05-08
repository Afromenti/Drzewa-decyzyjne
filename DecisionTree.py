
import numpy as np

from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def is_leaf_node(self):
    return self.value is not None
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
    def fit(self, x, y):
        self.n_features = x.shape[1] if not self.n_features else min(self.n_features, x.shape[1])
        self.root = self._grow_tree(x, y)
    def _grow_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(x, y, feat_idxs)
        left_idxs, right_idxs = self._split(x[:, best_feat], best_thresh)

        left = self._grow_tree(x[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(x[right_idxs], y[right_idxs], depth+1)

        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            vals = X[:, i]
            thresholds = np.unique(vals)
            for threshold in thresholds:
                gain = self._information_gain(y, vals, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_thresh = threshold
            return split_idx, split_thresh
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        if n_l == 0 or n_r == 0:
            return 0
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        w_l, w_r = n_l / n, n_r / n

        child_entropy = w_l * e_l + w_r * e_r
        ig = parent_entropy - child_entropy
        return ig
    def _split(self, X_column, split_thresh):
         left = np.argwhere(X_column <= split_thresh).flatten()
         right = np.argwhere(X_column > split_thresh).flatten()
         return left, right
    def _entropy(self, y):
        hist = np.bincount (y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
