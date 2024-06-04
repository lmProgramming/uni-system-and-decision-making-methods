from typing import List, Tuple
import numpy as np

def zeroes(shape: Tuple[int, int]):
    '''
    Generate list of lists of zeroes of shape (x, y)
    
    :return: array of zeroes
    '''
    x, y = shape
    
    return [[0] * x for _ in range(y)]


def get_confusion_matrix(
    y_true: List[int], y_pred: List[int], num_classes: int,
) -> List[List[int]]:
    """
    Generate a confusion matrix in a form of a list of lists. 

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values
    :param num_classes: number of supported classes

    :return: confusion matrix
    """
    
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("Invalid input shapes!")
    if max(y_true) >= num_classes or max(y_pred) >= num_classes:
        raise ValueError("Invalid prediction classes!")
    
    shape = (num_classes, num_classes)
    
    mat = zeroes(shape)
    
    for k in range(n):
        x = y_true[k]
        y = y_pred[k]
        mat[x][y] += 1   
        
    return mat


def get_quality_factors(
    y_true: List[int],
    y_pred: List[int],
) -> Tuple[int, int, int, int]:
    """
    Calculate True Negative, False Positive, False Negative and True Positive 
    metrics basing on the ground truth and predicted lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: a tuple of TN, FP, FN, TP
    """
    confusion_matrix = get_confusion_matrix(y_true, y_pred, 2)
    
    return confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the accuracy for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: accuracy score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    
    return (TN + TP) / len(y_true)


def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the precision for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: precision score
    """
    _, FP, _, TP = get_quality_factors(y_true, y_pred)
    
    return TP / (TP + FP)


def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the recall for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: recall score
    """
    
    _, _, FN, TP = get_quality_factors(y_true, y_pred)
    
    return TP / (TP + FN)


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the F1-score for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: F1-score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1
