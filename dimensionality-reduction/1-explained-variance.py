import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df_wine = pd.read_csv('../missing-data/wine.data', header=None)
df_wine.columns = [
    'Class label',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]

X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

std_sc = StandardScaler()
X_train_std = std_sc.fit_transform(X_train)
X_test_std = std_sc.transform(X_test)

# lr1 = LogisticRegression(penalty='l1', C=0.1)
# lr1.fit(X_train_std, y_train)
#
# feat_labels = df_wine.columns[1:]
# forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# forest.fit(X_train, y_train)
#
# X_selected = forest.transform(X_train, threshold=0.15)

# constructing the covariance matrix

cov_mat = np.cov(X_train_std.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
# print('---------------')
# print('eigen values: ')
# print(eigen_values)

# eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
# print('---------------')
# print('eigenh values: ')
# print(eigen_values)

tot = sum(eigen_values)
var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual exaplined variance')
# plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()

eigen_paris = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
eigen_paris.sort(reverse=True)

# print('eigen pairs: ')
# print(eigen_paris)

w = np.hstack((eigen_paris[0][1][:, np.newaxis], eigen_paris[1][1][:, np.newaxis]))
print('Matrix W: ')
print(w)

X_train_std[0].dot(w)

X_train_pca = X_train_std.dot(w)
# print('X train pca: ')
# print(X_train_pca)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
