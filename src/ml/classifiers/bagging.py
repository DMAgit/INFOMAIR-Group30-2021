from sklearn.ensemble import BaggingClassifier

from classifier import Classifier


class MyBaggingClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = BaggingClassifier()

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "bagging"