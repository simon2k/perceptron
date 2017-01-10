import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basic.perceptron import *
from plotter import *

df = pd.read_csv('./iris-data.csv', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.title('setosa vs versicolor')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker=0)
plt.xlabel('Epochs')
plt.ylabel('Number of misses')
plt.show()

plotter = Plotter(xlabel='sepal length [cm]', ylabel='petal length [cm]', legend_loc='upper left')
plotter.plot_decision_regions(X, y, classifier=ppn)
