from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


class Classifier:

    def save_to_file(self):
        pass

    @staticmethod
    def get_name():
        return "base-name"

    @staticmethod
    def get_file_name():
        return "base-name"

    def load_from_file(self):
        return self

    def set_grid_search_cv(self, params, x_train, y_train):
        pass

    @staticmethod
    def apply_bow(features_train, features_test):
        """
        Applies Bag of Words to the inputs and returns the corresponding matrices

        :param features_train: str - list of text features to transform (training)
        :param features_test: str - list of text features to transform (testing)
        :return: train, test: array - matrices associated to each vectorization
                    bow: CountVectorizer - the vectorization model used (for further reuse)
        """
        bow = CountVectorizer()
        train = bow.fit_transform(features_train)
        test = bow.transform(features_test)
        return train, test, bow
