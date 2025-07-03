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