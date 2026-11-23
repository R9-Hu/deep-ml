import numpy as np
def precision(y_true, y_pred):
    # Your code here
    TP = sum(1 if y_true[i] == y_pred[i] and y_true[i]==1 else 0 for i in range(len(y_true)))
    FP = sum(1 if y_true[i] != y_pred[i] and y_true[i]==0 else 0 for i in range(len(y_true)))
    return TP/(TP+FP+1e-10)
