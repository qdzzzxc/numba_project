import numpy as np
from collections import Counter
from numba import njit


class TreeNode:
    def __init__(
        self, data, feature_idx, feature_val, prediction_probs, information_gain
    ) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None

    def node_def(self) -> str:
        if self.left or self.right:
            return f"NODE | Information Gain = {self.information_gain} | Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right"
        else:
            unique_values, value_counts = np.unique(
                self.data[:, -1], return_counts=True
            )
            output = ", ".join(
                [
                    f"{value}->{count}"
                    for value, count in zip(unique_values, value_counts)
                ]
            )
            return (
                f"LEAF | Label Counts = {output} | Pred Probs = {self.prediction_probs}"
            )


class DecisionTree:
    """
    Decision Tree Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 1,
        min_information_gain: float = 0.0,
        numb_of_features_splitting: str = None,
    ) -> None:
        """
        Setting the class with hyperparameters
        max_depth: (int) -> max depth of the tree
        min_samples_leaf: (int) -> min
        min_information_gain: (float) -> min information gain required to make the splitting possible
        num_of_features_splitting: (str) ->  when splitting if sqrt then sqrt(
                                                            if log then log(
                                                            else all features are considered
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting

    @staticmethod
    @njit
    def _entropy(class_probabilities: np.ndarray) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p > 0])

    @staticmethod
    @njit
    def _find_label_probs(data: np.array, labels_in_train: np.array) -> np.array:
        labels_as_integers = data[:, -1].astype(np.int32)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(labels_in_train), dtype=float)

        for i in range(len(labels_in_train)):
            count = 0
            for j in range(total_labels):
                if labels_as_integers[j] == i:
                    count += 1
            label_probabilities[i] = count / total_labels

        return label_probabilities

    @staticmethod
    @njit
    def _find_best_split(data: np.array) -> tuple:
        """
        Finds the best split (with the lowest entropy) given data
        Returns 2 splitted groups and split information
        """
        min_part_entropy = 1e9

        feature_idx = np.arange(data.shape[1] - 1)
        feature_idx_to_use = np.random.choice(
            feature_idx, size=int(np.sqrt(len(feature_idx)))
        )

        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
            for feature_val in feature_vals:
                mask_below_threshold = data[:, idx] < feature_val
                g1 = data[mask_below_threshold]
                g2 = data[~mask_below_threshold]
                subsets = [g1[:, -1], g2[:, -1]]
                total_count = sum([len(subset) for subset in subsets])
                part_entropy = 0.0
                for subset in subsets:
                    total_count_subset = len(subset)
                    unique_values = np.unique(subset)
                    class_probabilities = np.zeros(shape=unique_values.shape)

                    for i, value in enumerate(unique_values):
                        class_probabilities[i] = (
                            np.sum(subset == value) / total_count_subset
                        )

                    subset_entropy = 0.0
                    for p in class_probabilities:
                        if p > 0:
                            subset_entropy -= p * np.log2(p)

                    part_entropy += subset_entropy * (total_count_subset / total_count)

                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        return (
            g1_min,
            g2_min,
            min_entropy_feature_idx,
            min_entropy_feature_val,
            min_part_entropy,
        )

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursive, depth first tree creation algorithm
        """

        if current_depth > self.max_depth:
            return None

        (
            split_1_data,
            split_2_data,
            split_feature_idx,
            split_feature_val,
            split_entropy,
        ) = self._find_best_split(data)

        label_probabilities = self._find_label_probs(data, self.labels_in_train)

        node_entropy = self._entropy(np.array(label_probabilities))
        information_gain = node_entropy - split_entropy

        node = TreeNode(
            data,
            split_feature_idx,
            split_feature_val,
            label_probabilities,
            information_gain,
        )

        if (
            self.min_samples_leaf > split_1_data.shape[0]
            or self.min_samples_leaf > split_2_data.shape[0]
        ):
            return node

        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)

        return node

    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets
        """

        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        self.tree = self._create_tree(data=train_data, current_depth=0)

        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)

        self.feature_importances = {
            k: v / total
            for total in (sum(self.feature_importances.values()),)
            for k, v in self.feature_importances.items()
        }

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)

        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)

        return preds

    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print("    " * 4 * level + "-> " + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        """Calculates the feature importance by visiting each node in the tree recursively"""
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)
