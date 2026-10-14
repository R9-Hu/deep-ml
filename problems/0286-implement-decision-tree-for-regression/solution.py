import numpy as np


def decision_tree_regressor(
    X_train,
    y_train,
    X_test,
    max_depth=2,
    min_samples_split=2
):
    """
    Build a decision tree for regression and predict on test data.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    
    def mse(y):
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def split(X, y, depth):
        n_samples, n_features = X.shape

        # Create a leaf node.
        if (
            depth == max_depth
            or n_samples < min_samples_split
            or np.all(y == y[0])
        ):
            return {
                "value": float(np.mean(y))
            }

        parent_error = mse(y)

        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        # Search every feature.
        for i in range(n_features):
            # order contains original row indices.
            order = np.argsort(X[:, i], kind="stable")

            # Apply the same permutation to X and y.
            X_copy = X[order]
            y_copy = y[order]

            # Check every boundary between consecutive values.
            for j in range(1, n_samples):
                # Equal values cannot be separated by a threshold.
                if X_copy[j, i] == X_copy[j - 1, i]:
                    continue

                left_count = j
                right_count = n_samples - j

                weighted_child_error = (
                    left_count / n_samples * mse(y_copy[:j])
                    + right_count / n_samples * mse(y_copy[j:])
                )

                gain_tmp = parent_error - weighted_child_error

                if gain_tmp > best_gain and not np.isclose(
                    gain_tmp,
                    best_gain,
                    rtol=1e-12,
                    atol=1e-12
                ):
                    best_gain = gain_tmp
                    best_feature = i
                    best_threshold = (
                        X_copy[j - 1, i] + X_copy[j, i]
                    ) / 2

                    # These are indices into the current X and y.
                    best_left_indices = order[:j]
                    best_right_indices = order[j:]

        # No feature can improve the prediction.
        if best_feature is None:
            return {
                "value": float(np.mean(y))
            }

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": split(
                X[best_left_indices],
                y[best_left_indices],
                depth + 1
            ),
            "right": split(
                X[best_right_indices],
                y[best_right_indices],
                depth + 1
            )
        }

    tree = split(X_train, y_train, depth=0)

    def predict(row, node):
        if "value" in node:
            return node["value"]

        feature = node["feature"]
        threshold = node["threshold"]

        if row[feature] <= threshold:
            return predict(row, node["left"])

        return predict(row, node["right"])

    return [
        round(predict(row, tree), 4)
        for row in X_test
    ]