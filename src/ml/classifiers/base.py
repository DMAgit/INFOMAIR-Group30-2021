from sklearn.dummy import DummyClassifier

from src.ml.classifiers.classifier import Classifier


class BaseClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = DummyClassifier(strategy="most_frequent")

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "base"
