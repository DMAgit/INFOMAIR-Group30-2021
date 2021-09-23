from joblib import dump, load
from sklearn.dummy import DummyClassifier

from classifier import Classifier


class BaseClassifier(Classifier):
    def __init__(self):
        self.model = DummyClassifier(strategy="most_frequent")
        pass

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "base"
