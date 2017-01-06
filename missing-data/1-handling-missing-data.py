import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
# print(df)
# print(df.isnull())
# print(df.isnull().sum())
# print(df.values)
#
# print('Removing values:')
# print(df.dropna())
# print()
# print(df.dropna(axis=1))
# print()
# print(df.dropna(how='all'))
# print('subset [C]:')
# print(df.dropna(subset=['C']))
# print('subset [D]:')
# print(df.dropna(subset=['D']))

# imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imr.fit(df)
# imputed_data = imr.transform(df.values)
# print('imputed data for column means:')
# print(imputed_data)
#
# imr = Imputer(missing_values='NaN', strategy='mean', axis=1)
# imr.fit(df)
# imputed_data = imr.transform(df.values)
# print('imputed data for row means:')
# print(imputed_data)
#
# imr = Imputer(missing_values='NaN', strategy='median', axis=1)
# imr.fit(df)
# imputed_data = imr.transform(df.values)
# print('imputed data median for row means:')
# print(imputed_data)
#
# imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
# imr.fit(df)
# imputed_data = imr.transform(df.values)
# print('imputed data most_frequent for row means:')
# print(imputed_data)

# imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
# imputed_data = imr.fit_transform(df.values)
# print('imputed data most_frequent for row means:')
# print(imputed_data)
