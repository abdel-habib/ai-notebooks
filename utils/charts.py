import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import io

def plot_radar_chart(df, title="", color_palette = ['#BAE8E1', '#B0BFFE', '#FFDFFC', '#FF9EB3']):
    '''
    Plots a radar chart.

    Args:
        - df (pd.DataFrame): A dataframe that its rows will be the radar circular points (metrics), and columns are the legends.

    Returns:
        Plot figure.
    '''
    # Get the metrics (df rows)
    metrics = df.index.tolist()

    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    
    # Title
    ax.set_title(title, y=1.15, fontsize=20) if title != "" else None

    # Direction of the zero angle to the north (upwards)
    ax.set_theta_zero_location("N")

    # Direction of the angles to be counterclockwise
    ax.set_theta_direction(-1)

    # Radial label position (position of values on the radial axes)
    ax.set_rlabel_position(90)

    # Make radial gridlines appear behind other elements
    ax.spines['polar'].set_zorder(1)

    # Color of radial girdlines
    ax.spines['polar'].set_color('lightgrey')

    
    for idx, (series_name, series) in enumerate(df.items()):
        values = series.tolist()
        values = values + [values[0]]

        # plots the lines
        ax.plot(theta, values, linewidth=1.75, linestyle='solid', label=series_name, marker='o', markersize=7, color=color_palette[idx % len(color_palette)])
    
        # This fills the areas
        ax.fill(theta, values, alpha=0.50, color=color_palette[idx % len(color_palette)], zorder=0)

    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"], color="black", size=11, zorder=10)
    plt.xticks(theta, metrics + [metrics[0]], color='black', size=11.5, zorder=10)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11.5)

def plot_boxplot(df, x_colname, y_colname, title="", hue_colname=None, color_palette = 'viridis'):
    '''
    Plots a boxplot.

    Args:
        - df (pd.DataFrame): A dataframe containing the data to plot.
        - x_colname (str): The column name for the x-axis.
        - y_colname (str): The column name for the y-axis (metric column).
        - title (str): The title of the plot.
        - hue_colname (str, optional): The column name for hue grouping. Default is None.
        - color_palette (list or str): The color palette to use for the plot.

    Returns:
        Plot figure.
    
    '''

    # Determine hue order and corresponding colors 
    hue_order = df[hue_colname].unique().tolist() if hue_colname else None
    color_palette = sns.color_palette(color_palette, len(hue_order)) if hue_order else None
    dataset_colormap = dict(zip(hue_order, color_palette)) if hue_order else None

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x=x_colname, y=y_colname, hue=hue_colname, palette=dataset_colormap, hue_order=hue_order)
    plt.title(title, fontsize=20) if title != "" else plt.title(y_colname.upper() + " distribution for " + x_colname.capitalize() + " and " + hue_colname.capitalize(), fontsize=20)
    plt.xlabel(x_colname.capitalize())
    plt.ylabel(y_colname.capitalize())
    plt.legend(title=hue_colname.capitalize(), bbox_to_anchor=(1.05, 1), loc="upper left") if hue_colname else None
    plt.tight_layout()
    plt.show()
