from matplotlib import pyplot as plt
import pandas as pd


def plot_pca(pca_df: pd.DataFrame, labels: list):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Первая главная компонента', fontsize = 15)
    ax.set_ylabel('Вторая главная компонента', fontsize = 15)
    ax.set_title('Метод главных компонент', fontsize = 20)

    colors = ['r', 'g', 'b', 'yellow', 'purple', 'gray']
    for target, color in zip(labels, colors):
        indicesToKeep = pca_df['Атака'] == target
        ax.scatter(pca_df.loc[indicesToKeep, pca_df.columns.values[0]]
                ,pca_df.loc[indicesToKeep, pca_df.columns.values[1]]
                ,c = color
                ,s = 50
                ,edgecolors='black')
    ax.legend(labels)
    ax.grid()

    plt.show()