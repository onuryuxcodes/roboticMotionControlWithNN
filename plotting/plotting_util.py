import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def line_plot_with_seaborn(dataframe, col_names_list, hue=None):
    sns.lineplot(data=dataframe[col_names_list], hue=hue)
    plt.show()


def scatter_plot_3d(e1, e2, v):
    plot_axes = plt.axes(projection='3d')
    plot_axes.scatter3D(e1, e2, v)
    plot_axes.set_xlabel('e1')
    plot_axes.set_ylabel('e2')
    plot_axes.set_zlabel('V')
    plt.show()



