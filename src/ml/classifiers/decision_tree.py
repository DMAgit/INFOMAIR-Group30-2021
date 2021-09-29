from sklearn.tree import DecisionTreeClassifier

from src.ml.classifiers.classifier import Classifier


class MyDecisionTreeClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "tree"