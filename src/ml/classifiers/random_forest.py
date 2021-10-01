from sklearn.ensemble import RandomForestClassifier

from classifier import Classifier


class MyRandomForestClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "randomforest"