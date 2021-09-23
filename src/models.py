from joblib import dump
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from rule_based import RuleBasedClassifier
from src.base import BaseClassifier
from src.complement_nb import ComplementNBClassifier
from src.sgdc_classifier import MySGDClassifier


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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=8)

    return x_train, x_test, y_train, y_test


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


def get_report(truth, predicted):
    """
    Returns the corresponding scores according to several metrics (such as accuracy, recall or precision) for each
    class, it also includes support and weighted/macro averages

    :param truth: array - vector containing the expect labels for each class
    :param predicted: array - vector containing the predicted labels for each class
    :return: dict/str - the classification report containing all the information and metrics
    """
    return classification_report(truth, predicted)


def get_classifiers(input_dimension=0):
    """
    Returns the classifiers declared as follows:
        [[model instance, *dict* with params to be tested], ...]
        - if the params dict is empty it will just perform a normal fit without cross-validation and default params

    :param input_dimension: int - the size of the training set (used to setup nn-models)
    :return: models: list - all the models with the different configurations
    """
    models = [
        # Baseline (majority class)
        [BaseClassifier(), {}],
        # Baseline (rule-based)
        [RuleBasedClassifier(), {}],
        # Other models
        [ComplementNBClassifier(),
            {"alpha": [0.1, 0.2, 0.4, 0.6, 0.8, 1]}],
        [MySGDClassifier(),
            {"alpha": 10.0 ** -np.arange(1, 7)}]
    ]

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
    raw_train, raw_test, y_train, y_test = load_dataset()
    x_train, x_test, bow = apply_bow(raw_train, raw_test)

    for i, classifier in enumerate(get_classifiers(x_train.shape[1])):  # used for defining nn-models (input_dim)
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
        dump(bow, "../models/bow.joblib")
