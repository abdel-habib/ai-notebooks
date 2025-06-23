import matplotlib.pyplot as plt

def linear_plot(x, y, title=''):
    """
    Plots a linear graph of x vs y.

    Parameters:
    x (list or array-like): x-axis values.
    y (list or array-like): y-axis values.
    title (str): Title of the plot.

    Returns:
    None
    """
    # plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.grid(True)
    plt.show()