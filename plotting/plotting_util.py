import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def line_plot_with_seaborn(dataframe, col_names_list, hue=None):
    sns.lineplot(data=dataframe[col_names_list], hue=hue)
    plt.show()



