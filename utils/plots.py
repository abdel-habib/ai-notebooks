import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()



def logistic_plot(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors='#0096ff', lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

def draw_logistic_vthresh(ax,x):
    """ draws a threshold """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color='#0096ff')
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color='#C00000')
    ax.annotate("z >= 0", xy= [x,0.5], xycoords='data',
                xytext=[30,5],textcoords='offset points')
    d = FancyArrowPatch(
        posA=(x, 0.5), posB=(x+3, 0.5), color='#C00000',
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(d)
    ax.annotate("z < 0", xy= [x,0.5], xycoords='data',
                 xytext=[-50,5],textcoords='offset points', ha='left')
    f = FancyArrowPatch(
        posA=(x, 0.5), posB=(x-3, 0.5), color='#0096ff',
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(f)


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
