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


def plot_loss(loss_list):
    epoch = [x for x in range(1, len(loss_list)+1)]
    plt.plot(epoch, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss.png')
    plt.clf()


def plot_data_point_num(inv_data_point_size_list, data_point_size_list):
    epoch = [x for x in range(1, len(data_point_size_list)+1)]
    max_data_p = max(data_point_size_list)
    plt.ylim(0, max_data_p+20)
    plt.plot(epoch, data_point_size_list, label="total data points", color='blue')
    plt.plot(epoch, inv_data_point_size_list, label="invalid data points", color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Data Point Size")
    plt.savefig('data_point_size.png')
    plt.clf()


def plot_invalid_data_point_num(inv_data_point_size_list):
    epoch = [x for x in range(1, len(inv_data_point_size_list)+1)]
    plt.plot(epoch, inv_data_point_size_list)
    plt.xlabel("Epoch")
    plt.ylabel("Invalid Data Point Size")
    plt.savefig('inv_data_point_size.png')
    plt.clf()




