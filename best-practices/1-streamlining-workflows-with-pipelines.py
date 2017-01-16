import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./wdbc.data.txt', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

# print(le.transform(['M', 'B']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# print(X_train)
# print(y_train)

pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', LogisticRegression(random_state=1))
])

# print("x train:")
# print(X_train)
#
# print("y train:")
# print(y_train)

pipe_lr.fit(X_train, y_train)
print("Test accuracy: {accuracy}".format(accuracy=round(pipe_lr.score(X_train, y_train), 3)))
print("Test accuracy: {accuracy}".format(accuracy=round(pipe_lr.score(X_test, y_test), 3)))

