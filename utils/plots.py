import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_similarity_heatmap(logits, image_labels, text_labels):
    """Plots a heatmap of image-text similarity logits.
    Parameters:
    logits (torch.Tensor): A tensor of shape (num_images, num_texts) containing
                            the similarity logits between images and text descriptions.
    image_labels (list): A list of labels for the images.
    text_labels (list): A list of labels for the text descriptions.

    Returns:
    None
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        logits.cpu().numpy(),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=text_labels,
        yticklabels=image_labels,
        cbar=True
    )
    plt.xlabel("Text Descriptions")
    plt.ylabel("Images")
    plt.title("Image–Text Similarity (Logits per Image)")
    plt.tight_layout()
    plt.show()
