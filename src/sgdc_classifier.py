from sklearn.linear_model import SGDClassifier

from classifier import Classifier


class MySGDClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = SGDClassifier(n_jobs=-1, max_iter=1000)

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "sgd"
