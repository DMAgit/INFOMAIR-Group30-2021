from sklearn.base import BaseEstimator, ClassifierMixin
from rule_dict import rule_dict


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting needed, rule implementation is done semi-manually
        return self

    def predict(self, X, y=None):
        return list(map(self.predict_most_likely, X))

    def predict_proba(self, X, y=None):
        return list(map(self.predict_most_likely, X))

    def predict_most_likely(self, word):
        words = word.split()
        for word in words:
            if rule_dict[word]:
                return rule_dict[word]
        return "inform"
