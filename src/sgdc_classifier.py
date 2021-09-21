from joblib import dump, load
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from classifier import Classifier


class MySGDClassifier(Classifier):
    def __init__(self):
        self.model = SGDClassifier(n_jobs=-1, max_iter=1000)
        pass

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def save_to_file(self):
        dump(self, self.get_file_name())

    def get_file_name(self):
        return "../models/sgd.joblib"

    def load_from_file(self):
        return load(self.get_file_name())

    def get_name(self):
        return "complementNB"

    def set_grid_search_cv(self, params, x_train, y_train):
        grid = GridSearchCV(estimator=self.model, param_grid=params, cv=10, n_jobs=-1,
                            scoring="f1_macro")
        grid.fit(x_train, y_train)
        self.model = grid.best_estimator_
        return self
