import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# print('X_train: ')
# print(X_train)
# print('X_test: ')
# print(X_test)

# print('y_train: ')
# print(y_train)
# print('y_test: ')
# print(y_test)

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

print('X_train_norm: ')
print(X_train_norm)
# print('X_test_norm: ')
# print(X_test_norm)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print('X_train_std: ')
print(X_train_std)
print('X_test_std: ')
print(X_test_std)
