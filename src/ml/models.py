import cleantext
import numpy as np
from joblib import dump
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.ml.classifiers.bagging import MyBaggingClassifier
from src.ml.classifiers.decision_tree import MyDecisionTreeClassifier
from src.ml.classifiers.random_forest import MyRandomForestClassifier
from src.ml.classifiers.rule_based import RuleBasedClassifier
from src.ml.classifiers.base import BaseClassifier
from src.ml.classifiers.complement_nb import MyComplementNBClassifier
from src.ml.classifiers.sgd import MySGDClassifier

from nltk.stem import PorterStemmer

pst = PorterStemmer()


def load_dataset():
    """
    Loads the dataset, performs basic line cleanup ("\n" removal) and then returns the set of features and labels for
    both training and testing

    :return: x_train, x_test: array - contains the features for the training and test sets
                y_train, y_test: array - contains the labels for the training and test sets
    """
    raw_data = dict()
    with open("../data/dialog_acts.dat", "r") as f:
        for line in f:
            [key, value] = line.split(" ", 1)
            value = value.strip("\n")
            if key not in raw_data:
                raw_data[key] = [value]
            else:  # to avoid duplicates
                if value not in raw_data[key]:
                    raw_data[key].append(value)

    # Prepare X and y variables
    x = []
    y = []
    for key in raw_data:
        for statement in raw_data[key]:
            x.append(statement)
            y.append(key)

    # Split train/test 85/15
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=88)

    return x_train, x_test, y_train, y_test


def preprocess_text(text):
    """
    Given a string cleans it up and returns it once processed. Applies:

    1) Lowercase
    2) Punctuation removal
    3) Normalize whitespaces
    4) Stem words using a semi-aggressive stemmer (the classical Porter Stemmer)

    :param text: str - the string to be processed
    :return: str - the processed string
    """
    # Lowercase, remove punctuation, and normalize whitespaces
    processed = cleantext.clean(text, lower=True, fix_unicode=True, no_punct=True)

    # Stem words
    try:
        stemmed_words = [pst.stem(word) for word in word_tokenize(processed)]
    except LookupError:
        import nltk
        nltk.download("punkt")
        stemmed_words = [pst.stem(word) for word in word_tokenize(processed)]

    return " ".join(stemmed_words)


def apply_tfidf(features_train, features_test):
    """
    Applies TFIDF to the inputs and returns the corresponding matrices

    :param features_train: str - list of text features to transform (training)
    :param features_test: str - list of text features to transform (testing)
    :return: train, test: array - matrices associated to each vectorization
                tfidf: TfidfTransformer - the vectorization model used (for further reuse)
    """
    bow = CountVectorizer(ngram_range=(1, 3))
    tfidf = TfidfTransformer()
    train = tfidf.fit_transform(bow.fit_transform(features_train))
    test = tfidf.transform(bow.transform(features_test))
    return train, test, tfidf


def get_report(truth, predicted):
    """
    Returns the corresponding scores according to several metrics (such as accuracy, recall or precision) for each
    class, it also includes support and weighted/macro averages

    :param truth: array - vector containing the expect labels for each class
    :param predicted: array - vector containing the predicted labels for each class
    :return: dict/str - the classification report containing all the information and metrics
    """
    return classification_report(truth, predicted)


def get_classifiers():
    """
    Returns the classifiers declared as follows:
        [[model instance, *dict* with params to be tested], ...]
        - if the params dict is empty it will just perform a normal fit without cross-validation and default params

    :return: models: list - all the models with the different configurations
    """
    models = [
        # Baseline (majority class)
        [BaseClassifier(), {}],
        # Baseline (rule-based)
        [RuleBasedClassifier(), {}],
        # Other models
        [MyDecisionTreeClassifier(),
         {"criterion": ["gini", "entropy"], "splitter": ["best", "random"],
          "max_features": [None, "auto", "sqrt", "log2"], "random_state": [88], "class_weight": [None, "balanced"]}],
        [MyRandomForestClassifier(),
         {"criterion": ["gini", "entropy"], "oob_score": [True, False],
          "max_features": [None, "auto", "sqrt", "log2"], "random_state": [88],
          "class_weight": [None, "balanced", "balanced_subsample"], "n_jobs": [-1]}],
        [MyBaggingClassifier(),
         {"base_estimator": [DecisionTreeClassifier(), SGDClassifier()],
          "random_state": [88], "n_jobs": [-1]}],
        [MyComplementNBClassifier(),
         {"alpha": [0.1, 0.2, 0.4, 0.6, 0.8, 1], "fit_prior": [True, False], "norm": [True, False]}],
        [MySGDClassifier(), {"alpha": 10.0 ** -np.arange(1, 7), "penalty": ["l1", "l2", "elasticnet"],
                             # "loss": ["hinge", "huber", "log", "squared_hinge"],
                             "random_state": [8], "shuffle": [True, False],
                             "learning_rate": ["optimal", "invscaling", "adaptive"],
                             "eta0": [0.1, 0.2, 0.4, 0.6, 0.8, 1], "max_iter": [500, 1000, 2500]
                             }]]

    return models


def get_classifier_names():
    """
    Returns the names of all the classifiers defined

    :return: names: list<str> - an string array with all the names
    """
    names = []
    for classifier in get_classifiers():
        names.append(classifier[0].get_name())
    return names


def execute_ml_pipeline(enable_save):
    """
    Executes the whole ML pipeline:
        - loading the dataset
        - applies bag of words
        - executes several models defined in the 'get_classifiers' function
        - print the results report
        - saves (if 'enable_save' == True) the models to future use

    :param enable_save: bool - True to save the models for future use, False for testing purposes
    """
    x_train, x_test, y_train, y_test = load_dataset()
    raw_train, raw_test = x_train, x_test
    x_train = list(map(lambda text: preprocess_text(text), x_train))
    x_test = list(map(lambda text: preprocess_text(text), x_test))
    x_train, x_test, tfidf = apply_tfidf(x_train, x_test)

    for i, classifier in enumerate(get_classifiers()):  # used for defining nn-models (input_dim)
        print(f"Training {classifier[0].get_name()}...")
        if classifier[1]:
            clf = classifier[0].set_grid_search_cv(classifier[1], x_train, y_train)
        else:  # no params defined (default training)
            clf = classifier[0].fit(x_train, y_train)

        if type(classifier[0]).__name__ == "RuleBasedClassifier":  # test with the raw texts
            y_pred = clf.predict(raw_test)
        else:
            y_pred = clf.predict(x_test)
        print(get_report(y_test, y_pred))

        if enable_save:
            clf.save_to_file()

    if enable_save:
        dump(tfidf, "../models/tfidf.joblib")


execute_ml_pipeline(True)
