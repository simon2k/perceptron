import numpy as np
from matplotlib.colors import ListedColormap

class Plotter(object):
    def __init__(self, plt):
        self.plt = plt

    def plot_decision_regions(self, X, y, classifier, resolution=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        x1_min = X[:, 0].min() - 1
        x1_max = X[:, 0].max() + 1

        x2_min = X[:, 1].min() - 1
        x2_max = X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )

        Z = classifier.predict(
            np.array(
                [xx1.ravel(), xx2.ravel()]
            ).T
        )
        Z = Z.reshape(xx1.shape)
        self.plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

        self.plt.xlim(xx1.min(), xx1.max())
        self.plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            self.plt.scatter(
                x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.8,
                c=cmap(idx),
                marker=markers[idx],
                label=cl
            )
