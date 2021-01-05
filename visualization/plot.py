import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.ticker as ticker

sns.set_context("paper")
# # Set the font to be serif, rather than sans
sns.set(font="serif")
# # Make the background white, and specify the
# # specific font family
sns.set_style(
    "white", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]}
)
plt.tight_layout()


def create_dataframe(x_data, y_data, region_name):
    """creates a data frame from input data along with region name used for plotting a data frame later

    Arguments:
        x_data {np.array} -- array of data points on x axis
        y_data {np.array} -- array of data points on y axis
        region_name {string} -- name of the region to be added to the plot

    Returns:
        pandas.dataframe -- a dataframe  used in the next function for plotting
    """
    data = {"x": x_data, "y": y_data}
    df = pd.DataFrame(data=data)
    data = {"x": x_data, "y": y_data}
    df["region"] = region_name
    return df


def plot_df(
    data_frame,
    output_file_path,
    ylabel="Accuracy",
    xlabel="Validation Subset",
    disable_x_axis=False,
    step_size=1,
):
    """used to plot a dataframe having data points on x and y axis

    Arguments:
        data_frame {pandas.dataframe} -- dataframe having data points
        output_file_path {stringq} -- path that will be used to save the plot

    Keyword Arguments:
        ylabel {str} -- y axis label (default: {'Accuracy'})
        xlabel {str} -- x axis label (default: {'Validation Subset'})
        disable_x_axis {bool} -- flag when set to true the x axis will be disabled in the plotted graph (default: {False})
        step_size (int, optional): step size on x axis. Defaults to 1.
    """
    sns.pointplot(
        x="x", y="y", data=data_frame, hue="region", palette=sns.color_palette("deep"),
    )
    plt.xlabel(xlabel, fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.grid()
    plt.legend(loc="lower right", fontsize="large")
    plt.yticks(fontsize=11)

    if disable_x_axis:
        plt.xticks([])
    else:
        plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(step_size))
    plt.savefig(output_file_path, bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.cla()
    plt.close()


def plot_original_masked(
    x_data,
    original_result,
    masked_result,
    ylabel,
    xlabel,
    storage_parent_dir,
    disable_x_axis=True,
    file_name=None,
    step_size=1,
):
    """plots unpruned model's data point vs pruned model (original/masked)

    Args:
        x_data (np.array): data point to be plotted on x axis
        original_result (np.array): array of original model results on y axis
        masked_result (np.array): array of masked model results on y axis
        ylabel (string): text to be written as y axis label
        xlabel (string): text to be written as x axis label
        storage_parent_dir (string): parent directory used to save plots
        disable_x_axis (bool, optional): a flag to disable/enable x axis. Defaults to True.
        file_name (string, optional): name of the output saved image. Defaults to None.
        step_size (int, optional): step size on x axis. Defaults to 1.
    """
    dataframe_masked = create_dataframe(x_data, masked_result, "Masked Model")
    dataframe_original = create_dataframe(x_data, original_result, "Original Model")
    data_frame_all = pd.concat([dataframe_masked, dataframe_original])
    if file_name is None:
        file_name = "{}_{}.jpg".format(
            ylabel.strip().lower().replace(" ", "_"), "_originalvsmasked"
        )
    file_path = os.path.join(storage_parent_dir, file_name,)
    plot_df(
        data_frame_all,
        file_path,
        ylabel=ylabel,
        xlabel=xlabel,
        disable_x_axis=disable_x_axis,
        step_size=step_size,
    )


if __name__ == "__main__":
    import numpy as np

    data_frame = create_dataframe(
        [i for i in range(20)], [np.random.rand() * 100 for i in range(20)], ""
    )
    plot_df(
        data_frame,
        "test.jpg",
        ylabel="Batch Loss",
        xlabel="batch index",
        disable_x_axis=True,
    )
    plot_original_masked(
        [i for i in range(20)],
        [90 for _ in range(20)],
        [np.random.rand() * 100 for i in range(20)],
        storage_parent_dir="./",
        ylabel="Batch Loss",
        xlabel="batch index",
        disable_x_axis=True,
    )
