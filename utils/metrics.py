import numpy as np
from typing import Union

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Compute the cosine distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine distance between `v1` and `v2`.
    '''
    vecs = (v1, v2) if len(v1.shape) >= len(v2.shape) else (v2, v1)
    return 1 - np.dot(*vecs) / (
            np.linalg.norm(v1, axis=len(v1.shape)-1) *
            np.linalg.norm(v2, axis=len(v2.shape)-1)
    )

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Compute the cosine similarity between two vectors usin the cosine distance function.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity between `v1` and `v2`.
    '''
    return 1 - cosine_distance(v1, v2)

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Euclidean distance between `v1` and `v2`.
    """
    dist = v1 - v2
    return np.linalg.norm(dist, axis=len(dist.shape)-1)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE) between true and predicted values, also called L1 loss. 
    
    Used in regression tasks that calculates the average absolute differences between predicted values from a machine learning model and the actual target values. 
    Unlike Mean Squared Error (MSE), MAE does not square the differences, treating all errors with equal weight regardless of their magnitude, making it less sensitive to outliers.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE) between true and predicted values, also called L2 loss. 

    MSE quantifies the magnitude of the error between an algorithm prediction and an actual output by taking the average of the squared difference between the predictions and the target 
    values. It is useful for regression tasks, particularly when we want to penalize larger errors more heavily.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between true and predicted values.

    RMSE is the square root of the MSE providing an error measure in the same units as the target variable. This makes it easier to interpret compared to MSE as it returns an error value 
    in the same scale as the dependent variable. RMSE is important when we need an interpretable measure of error that maintains sensitivity to larger mistakes making it suitable for many 
    regression tasks.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Root Mean Squared Error.
    """
    return np.sqrt(mse(y_true, y_pred))