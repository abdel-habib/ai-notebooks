import matplotlib.pyplot as plt
import numpy as np
import io

def plot_radar_chart(data, title="", color_palette = ['#BAE8E1', '#B0BFFE', '#FFDFFC', '#FF9EB3']):
    '''
    Plots a radar chart.

    Args:
        - data (pd.DataFrame): A dataframe that its rows will be the radar circular points (metrics), and columns are the legends.

    Returns:
        Plot figure.
    '''
    # Get the metrics (df rows)
    metrics = data.index.tolist()

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

    
    for idx, (series_name, series) in enumerate(data.items()):
        values = series.tolist()
        values = values + [values[0]]

        # plots the lines
        ax.plot(theta, values, linewidth=1.75, linestyle='solid', label=series_name, marker='o', markersize=7, color=color_palette[idx % len(color_palette)])
    
        # This fills the areas
        ax.fill(theta, values, alpha=0.50, color=color_palette[idx % len(color_palette)], zorder=0)

    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"], color="black", size=11, zorder=10)
    plt.xticks(theta, metrics + [metrics[0]], color='black', size=11.5, zorder=10)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11.5)