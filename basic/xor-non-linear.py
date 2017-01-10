from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
# plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='1')
# plt.ylim(-3.0)
# plt.legend()
# plt.show()

def plot_decisions_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, cmap=cmap, marker=markers[idx], label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
svm.fit(X_xor, y_xor)

plot_decisions_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=100000.0)
svm.fit(X_xor, y_xor)

plot_decisions_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()


svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=100000.0)
svm.fit(X_xor, y_xor)

plot_decisions_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.5, C=100000.0)
svm.fit(X_xor, y_xor)

plot_decisions_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=1.0, C=100000.0)
svm.fit(X_xor, y_xor)

plot_decisions_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()