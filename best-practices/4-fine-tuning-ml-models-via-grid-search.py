from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

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

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_train, y_train)

print("-------------")
print("Best score: {best_score}".format(best_score=round(gs.best_score_, 3)))
print("Best params: {best_params}".format(best_params=gs.best_params_))

clf = gs.best_estimator_
clf.fit(X_train, y_train)
score = round(clf.score(X_test, y_test), 3)
print("Test accuracy: {score}".format(score=score))
