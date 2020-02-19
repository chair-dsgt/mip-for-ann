import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def create_dataframe(x_data, y_data, region_name):
    """creates a data frame from input data along with region name used for plotting a data frame later

    Arguments:
        x_data {np.array} -- array of data points on x axis
        y_data {np.array} -- array of data points on y axis
        region_name {string} -- name of the region to be added to the plot

    Returns:
        pandas.dataframe -- a dataframe  used in the next function for plotting
    """
    data = {'x': x_data, 'y': y_data}
    df = pd.DataFrame(data=data)
    data = {'x': x_data, 'y': y_data}
    df['region'] = region_name
    return df


def plot_df(data_frame, output_file_path, ylabel='Accuracy', xlabel='Validation Subset', disable_x_axis=False):
    """used to plot a dataframe having data points on x and y axis

    Arguments:
        data_frame {pandas.dataframe} -- dataframe having data points
        output_file_path {stringq} -- path that will be used to save the plot

    Keyword Arguments:
        ylabel {str} -- y axis label (default: {'Accuracy'})
        xlabel {str} -- x axis label (default: {'Validation Subset'})
        disable_x_axis {bool} -- flag when set to true the x axis will be disabled in the plotted graph (default: {False})
    """
    sns.pointplot(x='x', y='y', data=data_frame, hue='region')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.grid()
    plt.legend(loc='upper right')
    if disable_x_axis:
        plt.xticks([])
    plt.savefig(output_file_path)
    plt.clf()
    plt.cla()
    plt.close()
