import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df_wine = pd.read_csv('wine.data', header=None)
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

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

lr1 = LogisticRegression(penalty='l1', C=0.1)
lr1.fit(X_train_std, y_train)

# print('[L1] Training accuracy: ', lr1.score(X_train_std, y_train))
# print('[L1] Test accuracy: ', lr1.score(X_test_std, y_test))
# print('[L1] Intercept: ', lr1.intercept_)

lr2 = LogisticRegression(penalty='l2', C=0.1)
lr2.fit(X_train_std, y_train)

# print('---------------------------------------------------------------')
# print('[L2] Training accuracy: ', lr2.score(X_train_std, y_train))
# print('[L2] Test accuracy: ', lr2.score(X_test_std, y_test))
# print('[L2] Intercept: ', lr2.intercept_)

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# for f in range(X_train.shape[1]):
#     print(
#         "{index}) {label} \t {importance}".format(
#             index=f + 1,
#             label=feat_labels[indices[f]],
#             importance=importances[indices[f]]
#         )
#     )
#
# plt.title('Feature Importances')
# plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
# plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.show()

X_selected = forest.transform(X_train, threshold=0.15)
print(X_selected)
print(X_selected.shape)
