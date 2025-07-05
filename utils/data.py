import numpy as np

def load_house_data():
    """ 
    Load house data from a text file.

    Returns:
        X (numpy.ndarray): Features of the houses (size, number of rooms, etc.).
        y (numpy.ndarray): Target values (prices of the houses).
    """

    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y