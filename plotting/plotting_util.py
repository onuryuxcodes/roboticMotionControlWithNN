import seaborn as sns


def line_plot_with_seaborn(dataframe, x_col_name, y_col_name, hue=None):
    sns.lineplot(data=dataframe, x=x_col_name, y=y_col_name, hue=hue)


