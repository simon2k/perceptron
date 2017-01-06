import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1'],
])

df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

size_mapping = {'XL': 3, 'L': 2, 'M': 1}

df['size'] = df['size'].map(size_mapping)

# print('after mapping size ordinal feature to its value')
# print(df)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# print('class mapping: ')
# print(class_mapping)

# print('mapping class values in df:')
df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

# class_le = LabelEncoder()
# class_values = class_le.fit_transform(df['classlabel'].values)
# print('class values')
# print(class_values)
# orig = class_le.inverse_transform(class_values)
# print(orig)

X = df[['color', 'size', 'price']].values
# print('X: ')
# print(X)

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
# print('After adding value for color:')
# print(X)

ohe = OneHotEncoder(categorical_features=[0])
transformed = ohe.fit_transform(X)
print('one hot encoder: ')
print(transformed.toarray())

with_dummies = pd.get_dummies(df[['price', 'color', 'size']])
print('with dummies: ')
print(with_dummies)
