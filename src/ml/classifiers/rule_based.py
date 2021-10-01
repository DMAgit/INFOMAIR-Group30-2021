from sklearn.base import BaseEstimator, ClassifierMixin
from rule_dict import rule_dict, count_dict
from classifier import Classifier


class RuleBasedClassifier(BaseEstimator, ClassifierMixin, Classifier):
    def __init__(self):
        super().__init__()
        self.model = self

    def fit(self, x=None, y=None):
        # No fitting needed, rule implementation is done semi-manually
        return self

    def predict(self, x, y=None):
        return list(map(self.predict_most_likely, x))

    def transform_and_predict(self, sentence, tfidf):
        return "".join(self.predict([sentence]))

    def get_name(self):
        return "rulebased"

    @staticmethod
    def predict_most_likely(sentence):
        c_d_temp = count_dict.copy()
        for word in sentence.split():  # check over each word
            if word in rule_dict.keys():  # if it is a keyword
                c_d_temp[rule_dict.get(word)] += 1  # add one to that keyword
        if max(c_d_temp.values()) != 0:  # if any keyword was found
            return max(c_d_temp, key=c_d_temp.get)  # return the most common one
        else:
            return "inform"  # the most common class
