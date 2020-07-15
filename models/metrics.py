"""
This module contains the different metrics to evaluate the quality of predictions
sklearn anyway implements all of them
Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> pred = [int(np.round(i)) for i in np.random.rand(20)]
    >>> Y = [int(np.round(i)) for i in np.random.rand(20)]
    >>> _get_number_of_true_positive(pred, Y)
    11
    >>> _get_number_of_true_negative(pred, Y)
    4
    >>> _get_number_of_false_negative(pred, Y)
    2
    >>> _get_number_of_false_positive(pred, Y)
    3
    >>> get_recall(pred, Y)
    0.8461538461538461
    >>> get_accuracy(pred, Y)
    0.75
    >>> get_precision(pred, Y)
    0.7857142857142857
    >>> get_f1_score(pred, Y)
    0.8148148148148148
"""
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score

def _get_diagonal_elements(mat):
    return [mat[i][i] for i in range(len(mat))]


def get_accuracy(pred, y): # also np.sum(_get_diagonal_elements(confusion_matrix(pred, Y))) / len(Y)
    return (pred == y).mean()


def _get_f1_from_precision_and_recall(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def get_f1_score(pred, Y):
    return _get_f1_from_precision_and_recall(get_precision(pred, Y), get_recall(pred, Y))


def get_recall(pred, Y):
    """    Returns recall defined as: TP / (FN + TP) (true positive / condition positive)    """
    TP = _get_number_of_true_positive(pred, Y)
    FN = _get_number_of_false_negative(pred, Y)
    return TP / (TP + FN)


def get_precision(pred, Y):
    """    Returns precision defined as: TP / (TP + FP)    """
    TP = _get_number_of_true_positive(pred, Y)
    FP = _get_number_of_false_positive(pred, Y)
    return TP / (TP + FP)


# True/False - Positive/Negative (All of these can be also done with sklearn.metrics.confusion_matrix)
def _get_number_of_true_positive(pred, Y):
    # same as confusion_matrix[1][1]
    return np.sum([p and l for p, l in zip(np.array(pred) == 1, np.array(Y) == 1)])

def _get_number_of_true_negative(pred, Y):
    # same as confusion_matrix[0][0]
    return np.sum([p and l for p, l in zip(np.array(pred) == 0, np.array(Y) == 0)])

def _get_number_of_false_negative(pred, Y):
    # same as confusion_matrix[0][1]
    return np.sum([p and l for p, l in zip(np.array(pred) == 0, np.array(Y) == 1)])

def _get_number_of_false_positive(pred, Y):
    # same as confusion_matrix[1][0]
    return np.sum([p and l for p, l in zip(np.array(pred) == 1, np.array(Y) == 0)])

# performance
def _get_performance(model, X, Y):
    """ Returns the accuracy of the model """
    pred = model.predict(X)
    return accuracy_score(pred, Y)


def _get_full_performance(pred, y) -> list:
    """ Returns accuracy, precision, recall and f1-score of the model """
    accuracy = get_accuracy(pred, y)
    try:
        f1 = f1_score(pred, y, average='binary')
        recall = recall_score(pred, y, average='binary')
        precision = precision_score(pred, y, average='binary')
    except ValueError:
        f1 = f1_score(pred, y, average='weighted')
        recall = recall_score(pred, y, average='weighted')
        precision = precision_score(pred, y, average='weighted')
    return [accuracy, precision, recall, f1]


def _get_full_performance_df(model, X, Y) -> pd.DataFrame:
    """ Returns accuracy, precision, recall and f1-score of the model for each category in a DataFrame
    Returns:
        pd.DataFrame with columns=['accuracy', 'precision', 'recall', 'f1-score']
        and index the categories (columns) of Y
    """
    pred = model.predict(X)
    pred_df = pd.DataFrame(pred, columns=Y.columns, index=Y.index).astype(int)
    out = pd.DataFrame(index=Y.columns, columns=['accuracy', 'precision', 'recall', 'f1-score'])
    for cat in Y.columns:
        p = pred_df[cat].astype(int)
        y = Y.astype(int)[cat]
        out.loc[cat] = _get_full_performance(p, y)
    return out


def _print_performance(accuracy: pd.Series, verbosity='mean') -> None:
    """ Prints the accuracy value(s)
    Args:
        accuracy: accuracy of the predictor as a Series
        verbosity: if 'mean' will print only the mean value, otherwise, will print to screen each label's accuracy
    """
    if verbosity == 'mean':
        print("{:3.3f}".format(accuracy.mean()))
    else:
        print(accuracy.round(3))


def _print_full_performance(performance: pd.DataFrame, verbosity='mean') -> None:
    """ Prints the accuracy, precision, recall and f-1 score
    Args:
        accuracy: performance of the predictor as a DataFrame
        verbosity: if 'mean' will print only the mean value(s), otherwise, will print for each label
    """
    if verbosity == 'mean':
        print(performance.mean().astype(float).round(3))
    else:
        print(performance.astype(float).round(3))

