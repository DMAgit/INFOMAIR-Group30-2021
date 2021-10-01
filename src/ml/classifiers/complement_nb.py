from sklearn.naive_bayes import ComplementNB

from classifier import Classifier


class MyComplementNBClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = ComplementNB()

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "complementnb"
