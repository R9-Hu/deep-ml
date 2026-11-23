import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    act_positive = np.asarray(y_true) == 1
    n_predicted = np.count_nonzero(act_positive)
    if n_predicted == 0:
        return 0.0
    predicted_positive = np.asarray(y_pred) == 1
    tp = np.count_nonzero(predicted_positive & (np.asarray(y_true) == 1))
    return tp / n_predicted
