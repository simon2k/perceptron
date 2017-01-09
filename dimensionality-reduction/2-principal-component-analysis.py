from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from plotter import Plotter
from df_wine import DfWine

df_wine = DfWine()

pca = PCA(n_components=2)
lr = LogisticRegression()

X_train_pca = pca.fit_transform(df_wine.X_train_std)
X_test_pca = pca.transform(df_wine.X_test_std)

lr.fit(X_train_pca, df_wine.y_train)

train_plotter = Plotter(xlabel='PC1', ylabel='PC2', legend_loc='lower left')
train_plotter.plot_decision_regions(X_train_pca, df_wine.y_train, classifier=lr)

test_plotter = Plotter(xlabel='PC1', ylabel='PC2', legend_loc='lower left')
test_plotter.plot_decision_regions(X_test_pca, df_wine.y_test, classifier=lr)
