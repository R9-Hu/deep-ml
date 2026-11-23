import numpy as np
def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score based on true and predicted labels.

    Args:
        y_true (list): True labels (ground truth).
        y_pred (list): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    # Your code here
    act_positive = np.asarray(y_true) == 1
    n_act = np.count_nonzero(act_positive)
    if n_act == 0:
        return 0.0

    predicted_positive = np.asarray(y_pred) == 1
    n_predicted = np.count_nonzero(predicted_positive)
    if n_predicted == 0:
        return 0.0
    tp = np.count_nonzero(predicted_positive & (np.asarray(y_true) == 1))
    P = tp / n_predicted
    R = tp / n_act
    if P + R == 0:
        return 0.0
    f1 = 2*P*R/(P+R)

    return round(f1,3)