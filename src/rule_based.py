from sklearn.base import BaseEstimator, ClassifierMixin
from rule_dict import rule_dict
from rule_dict import count_dict


class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        # No fitting needed, rule implementation is done semi-manually
        return self

    def predict(self, x, y=None):
        return list(map(self.predict_most_likely, x))

    @staticmethod
    def predict_most_likely(sentence):
        c_d_temp = count_dict.copy()
        for word in sentence.split():  # check over each word
            if word in rule_dict.keys():  # if it is a keyword
                c_d_temp[rule_dict.get(word)] += 1  # add one to that keyword
        if max(c_d_temp.values()) != 0:  # if any keyword was found
            print(c_d_temp)
            print(max(c_d_temp, key=c_d_temp.get))
            return max(c_d_temp, key=c_d_temp.get)  # return the most common one
        else:
            print("None")
            return "inform"  # the most common class
