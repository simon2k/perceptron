from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plotter import Plotter
from df_wine import DfWine
import numpy as np
import matplotlib.pyplot as plt

df_wine = DfWine()

# pca = PCA(n_components=2)
# lr = LogisticRegression()
#
# X_train_pca = pca.fit_transform(df_wine.X_train_std)
# X_test_pca = pca.transform(df_wine.X_test_std)
#
# lr.fit(X_train_pca, df_wine.y_train)
#
# plotter = Plotter(xlabel='PC1', ylabel='PC2', legend_loc='lower left')
# plotter.plot_decision_regions(X_train_pca, df_wine.y_train, classifier=lr)
#
# plotter2 = Plotter(xlabel='PC1', ylabel='PC2', legend_loc='lower left')
# plotter2.plot_decision_regions(X_test_pca, df_wine.y_test, classifier=lr)

np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(df_wine.X_train_std[df_wine.y_train == label], axis=0))
    # print("MV {label}: ${means}".format(label=label, means=mean_vecs[label-1]))

features_number = 13
S_W = np.zeros((features_number, features_number))

for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((features_number, features_number))
    for row in df_wine.X_train[df_wine.y_train == label]:
        row = row.reshape(features_number, 1)
        mv = mv.reshape(features_number, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter

# print("\nNot scaled Within-class scatter matrix: {x}x{y}\n".format(x=S_W.shape[0], y=S_W.shape[1]))

ys = np.array(df_wine.y_train, dtype=np.int32)
distribution = np.bincount(ys)[1:]
# print("\nclass label distribution: {distribution}\n".format(distribution=distribution))

features_number = 13
S_W = np.zeros((features_number, features_number))

for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(df_wine.X_train_std[df_wine.y_train == label].T)
    S_W += class_scatter

# print("\nScaled within-class scatter matrix: {x}x{y}\n".format(x=S_W.shape[0], y=S_W.shape[1]))

mean_overall = np.mean(df_wine.X_train_std, axis=0)
S_B = np.zeros((features_number, features_number))

for i, mean_vec in enumerate(mean_vecs):
    n = df_wine.X_train[df_wine.y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(features_number, 1)
    mean_overall = mean_overall.reshape(features_number, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

# print("\nBetween-class scatter matrix: {x}x{y}\n".format(x=S_B.shape[0], y=S_B.shape[1]))

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# print("eigenvalues in decreasing order: \n")
# for eigen_val in eigen_pairs:
#     print(eigen_val[0])


total = sum(eigen_vals.real)
discr = [(i / total) for i in sorted(eigen_vals.real, reverse=True)]
cum_disc = np.cumsum(discr)

# plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminability"')
# plt.step(range(1, 14), cum_disc, where='mid', label='cumulative "discriminability"')
# plt.ylabel('"discriminability" ration')
# plt.xlabel('Linear discriminations')
# plt.legend(loc='best')
# plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
# print("matrix W: \n{w}".format(w=w))

# X_train_lda = df_wine.X_train_std.dot(w)
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
#
# for l, c, m in zip(np.unique(df_wine.y_train), colors, markers):
#     plt.scatter(
#         X_train_lda[df_wine.y_train == l, 0], # * (-1),
#         X_train_lda[df_wine.y_train == l, 1], # * (-1),
#         c=c, label=l, marker=m
#     )

# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower right')
# plt.show()

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(df_wine.X_train_std, df_wine.y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, df_wine.y_train)
train_plotter = Plotter(xlabel='LD 1', ylabel='LD 2', legend_loc='lower left')
train_plotter.plot_decision_regions(X_train_lda, df_wine.y_train, classifier=lr)

X_test_lda = lda.transform(df_wine.X_test_std)
test_plotter = Plotter(xlabel='LD 1', ylabel='LD 2', legend_loc='lower left')
test_plotter.plot_decision_regions(X_test_lda, df_wine.y_test, classifier=lr)
