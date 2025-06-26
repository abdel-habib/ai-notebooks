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

def plot_histograms(histograms = {}):
    """Plots histograms for the given data.
    
    Parameters:
    histograms (list): A list of dictionaries where keys are histogram names and values are lists of data to plot.

    Returns:
    None
    """
    num_plots = len(histograms)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    if num_plots == 1:
        axes = [axes]

    for ax, data in zip(axes, histograms):
        if not hasattr(data, 'values'):
            raise ValueError("Each histogram data must have a 'value' attribute.")

        ax.hist(data['values'], bins=30)
        ax.set_title(data['name'])

    plt.show()