import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class Monitor(object):
    def __init__(self):
        '''Monitor class to monitor training progress and plot relevant metrics.
        It stores training and validation losses, learning rates, epoch timestamps and performance metrics.

        Parameters
        ----------
        logs : dict
            A dictionary containing training and validation losses, learning rates, epoch timestamps,
            and performance metrics. Expected keys include:
            - 'loss_tr': list of training losses per epoch
            - 'loss_val': list of validation losses per epoch
            - 'lrs': list of learning rates per epoch
            - 'epoch_start_timestamps': list of epoch start timestamps
            - 'epoch_end_timestamps': list of epoch end timestamps
            - Additional performance metrics with keys following the format '<metric_name>__metric'
        '''
        self.logs = {
            'loss_tr': [],
            'loss_val': [],
            'lrs': [],
            'epoch_start_timestamps': [],
            'epoch_end_timestamps': []
            }

    def plot(
            self,
            export_path: Optional[str] = None,
        ) -> None:
        '''Plot training progress including losses, performance metrics, epoch durations, and learning rates.

        Parameters
        ----------
        export_path : str, optional
            If provided, the plot will be saved to this path as 'progress.png'.
            If not provided, the plot will be displayed inline (if supported) or not shown at all.

        Returns
        -------
        None 
        '''
        
        # Obtain the logging keys names
        required_keys = ["lrs", "epoch_start_timestamps", "epoch_end_timestamps"]
        loss_keys = [k for k in self.logs.keys() if "loss" in k]
        perf_keys = [k for k in self.logs.keys() if k.endswith("__metric")]

        # ----------- Validate Keys -----------
        assert all(k in self.logs.keys() for k in required_keys), f"missing required log keys: {required_keys} in the logs keys: {self.logs.keys()}"
        assert len(loss_keys) > 0, "mising loss keys in the logs dict, please include keys with correct naming format (i.e. 'train__losses')."
        assert len(perf_keys) > 0, "mising performance keys in the logs dict, please include keys with correct naming format (i.e. 'dice__metrics')."
        
        # Determine epoch count
        epoch = min(len(v) for v in self.logs.values()) - 1
        x_values = list(range(epoch + 1))

        sns.set(font_scale=2.5)
        fig, axes = plt.subplots(3, 1, figsize=(30, 54))

        # Plotting styles
        _styles = [
            {'linestyle': '-', 'color': 'b'},
            {'linestyle': '-', 'color': 'r'},
            {'linestyle': '--', 'color': 'g'},
            {'linestyle': 'dotted', 'color': 'g'},
        ]

        # ---------- Subplot 1: Loss + Performance Metrics ----------
        ax1 = axes[0]
        ax1b = ax1.twinx()

        # Loss curves
        for i, k in enumerate(loss_keys):
            ax1.plot(x_values, self.logs[k][:epoch+1], ls=_styles[i]['linestyle'], c=_styles[i]['color'], linewidth=4, label=k)

        # Perf metrics
        for i, k in enumerate(perf_keys):
            ax1b.plot(x_values, self.logs[k][:epoch+1], ls=_styles[i+2]['linestyle'], c=_styles[i+2]['color'], linewidth=4, label=k.replace("__metric", ""))

        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.legend(loc=(0,1))
        ax1b.set_ylabel("performance metrics")
        ax1b.legend(loc=(0.2,1))

        # ---------- Subplot 2: Epoch duration ----------
        ax2 = axes[1]
        durations = [
            self.logs['epoch_end_timestamps'][i] - self.logs['epoch_start_timestamps'][i]
            for i in range(epoch+1)
        ]
        ax2.plot(x_values, durations, linewidth=4, ls=_styles[0]['linestyle'], c=_styles[0]['color'], label="epoch duration")
        ax2.set_ylabel("time [s]")
        ax2.set_xlabel("epoch")
        ax2.legend(loc=(0,1))
        ax2.set_ylim([0, ax2.get_ylim()[1]])

        # ---------- Subplot 3: Learning rate ----------
        ax3 = axes[2]
        ax3.plot(x_values, self.logs['lrs'][:epoch+1], linewidth=4, ls=_styles[0]['linestyle'], c=_styles[0]['color'], label="learning rate")
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("learning rate")
        ax3.legend(loc=(0,1))

        plt.tight_layout()

        # --- Save or return ---
        if export_path:
            assert os.path.exists(export_path), f"export path {export_path} does not exist."
            os.makedirs(export_path, exist_ok=True)
            path = os.path.join(export_path, "progress.png")
            fig.savefig(path, dpi=200)
            print(f"saved monitor plot to {path}")

            plt.close(fig)  # prevent auto-display
            return None

        plt.show()
        
    def get_checkpoint(self):
        '''Get the current logs for checkpointing.'''
        return self.logs

    def load_checkpoint(self, checkpoint: dict):
        '''Load logs from a checkpoint to continue training.
        '''
        self.logs = checkpoint

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
