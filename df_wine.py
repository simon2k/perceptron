from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DfWine:
    def __init__(self):
        self.data = pd.read_csv('../missing-data/wine.data', header=None)
        self.columns = [
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

        self.X = self.data.iloc[:, 1:].values
        self.y = self.data.iloc[:, 0].values

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.3, random_state=0)

        self.std_sc = StandardScaler()
        self.X_train_std = self.std_sc.fit_transform(self.X_train)
        self.X_test_std = self.std_sc.transform(self.X_test)
