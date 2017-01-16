from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

pipe_svc = Pipeline([
    ('scl', StandardScaler()),
    ('clf', SVC(random_state=1))
])

df = pd.read_csv('./wdbc.data.txt', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

param_range = [.0001, .001, .01, .1, 1., 10., 100., 1000.]

param_grid = [
    {'clf__C': param_range, 'clf__kernel': ['linear']},
    {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}
]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=1)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)

print("-------------")
mean = round(np.mean(scores), 3)
std = round(np.std(scores), 3)
print("CV accuracy: {mean} +/- {std}".format(mean=mean, std=std))

gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[
        {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
    scoring='accuracy',
    cv=5)

scores = cross_val_score(gs,
                         X_train,
                         y_train,
                         scoring='accuracy',
                         cv=2)

print("-------------")
mean = round(np.mean(scores), 3)
std = round(np.std(scores), 3)
print("CV accuracy: {mean} +/- {std}".format(mean=mean, std=std))
