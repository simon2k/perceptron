import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

print('[L1] Training accuracy: ', lr1.score(X_train_std, y_train))
print('[L1] Test accuracy: ', lr1.score(X_test_std, y_test))
print('[L1] Intercept: ', lr1.intercept_)

lr2 = LogisticRegression(penalty='l2', C=0.1)
lr2.fit(X_train_std, y_train)

print('---------------------------------------------------------------')
print('[L2] Training accuracy: ', lr2.score(X_train_std, y_train))
print('[L2] Test accuracy: ', lr2.score(X_test_std, y_test))
print('[L2] Intercept: ', lr2.intercept_)

fig = plt.figure()
ax = plt.subplot(111)

colors = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo',
    'orange'
]

weights = []
params = []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)

plt.xlim([10**(-5), 10**5])
plt.ylabel('weights coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()
