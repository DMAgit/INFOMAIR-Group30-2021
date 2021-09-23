from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB

from classifier import Classifier


class ComplementNBClassifier(Classifier):
    def __init__(self):
        self.model = ComplementNB()
        pass

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "complementNB"

    def set_grid_search_cv(self, params, x_train, y_train):
        grid = GridSearchCV(estimator=self.model, param_grid=params, cv=10, n_jobs=-1,
                            scoring="f1_macro")
        grid.fit(x_train, y_train)
        self.model = grid.best_estimator_
        return self
