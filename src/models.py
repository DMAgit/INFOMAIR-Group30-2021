from joblib import dump
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from rule_based import RuleBasedClassifier
from nn_model import get_model


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
        [DummyClassifier(strategy="most_frequent"), {}],
        # Baseline (rule-based)
        [RuleBasedClassifier(), {}],
        # Other models
        [ComplementNB(),
            {"alpha": [0.1, 0.2, 0.4, 0.6, 0.8, 1]}],
        [SGDClassifier(n_jobs=-1, max_iter=1000),
            {"alpha": 10.0 ** -np.arange(1, 7)}],
        [get_model(input_dimension),  # Keras model
         {"epochs": [50, 100], "batch_size": [32, 64, 128]}]
    ]

    return models


def get_classifier_names():
    """
    Returns the names of all the classifiers defined

    :return: names: list<str> - an string array with all the names
    """
    names = []
    for classifier in get_classifiers():
        names.append(type(classifier[0]).__name__)
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
    raw_train, raw_test = x_train, x_test  # for the rule-based clf
    x_train, x_test, bow = apply_bow(x_train, x_test)

    clf = None

    for i, classifier in enumerate(get_classifiers(x_train.shape[1])):  # used for defining nn-models (input_dim)
        print("Training {}...".format(type(classifier[0]).__name__))
        if classifier[1]:
            grid = GridSearchCV(estimator=classifier[0], param_grid=classifier[1], cv=10, n_jobs=-1, scoring="f1_macro")
            grid.fit(x_train, y_train)
            clf = grid.best_estimator_
        else:
            if type(classifier[0]).__name__ == "RuleBasedClassifier":  # No need for training
                pass
            else:
                clf = classifier[0].fit(x_train, y_train)

        if type(classifier[0]).__name__ == "RuleBasedClassifier":  # Test with the raw texts
            y_pred = classifier[0].predict(raw_test)
        else:
            y_pred = clf.predict(x_test)
        print(get_report(y_test, y_pred))

        if enable_save and type(classifier[0]).__name__ != "KerasClassifier":
            dump(clf, "../models/ml{}.joblib".format(i))
        else:
            clf.model.save("../models/ml{}.h5".format(i))

    if enable_save:
        dump(bow, "../models/bow.joblib")
