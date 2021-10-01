from joblib import load, dump
from sklearn.model_selection import RandomizedSearchCV


class Classifier:
    def __init__(self):
        self.model = None
        pass

    def save_to_file(self):
        dump(self, self.get_file_name())

    @staticmethod
    def get_name():
        return "base-name"

    def get_file_name(self):
        return f"../models/{self.get_name()}.joblib"

    def load_from_file(self):
        return load(self.get_file_name())

    def predict(self, sentence):
        return []

    def transform_and_predict(self, sentence, tfidf):
        return "".join(self.predict(tfidf.transform([sentence])))

    def set_grid_search_cv(self, params, x_train, y_train):
        grid = RandomizedSearchCV(estimator=self.model, param_distributions=params, cv=10, n_jobs=-1,
                                  scoring="accuracy", refit=True, random_state=88)
        grid.fit(x_train, y_train)
        self.model = grid.best_estimator_
        return self
