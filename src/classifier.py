from joblib import load, dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


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

    def transform_and_predict(self, sentence, bow):
        return "".join(self.predict(bow.transform([sentence])))

    def set_grid_search_cv(self, params, x_train, y_train):
        grid = GridSearchCV(estimator=self.model, param_grid=params, cv=10, n_jobs=-1,
                            scoring="f1_macro")
        grid.fit(x_train, y_train)
        self.model = grid.best_estimator_
        return self

    @staticmethod
    def apply_bow(features_train, features_test):
        """
        Applies Bag of Words to the inputs and returns the corresponding matrices

        :param features_train: str - list of text features to transform (training)
        :param features_test: str - list of text features to transform (testing)
        :return: train, test: array - matrices associated to each vectorization
                    bow: CountVectorizer - the vectorization model used (for further reuse)
        """
        bow = CountVectorizer()
        train = bow.fit_transform(features_train)
        test = bow.transform(features_test)
        return train, test, bow
