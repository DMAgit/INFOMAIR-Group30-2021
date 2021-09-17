from joblib import load
from tensorflow import keras
from models import execute_ml_pipeline, get_classifier_names
import argparse
import glob

""" Entry point to the application
"""


def main():
    """
    Entry point of application, writes instructions and takes input.
    """
    # Add parser for flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', action='store_true')

    isDeveloper = parser.parse_args().development

    clf_str_available, clf_str_input, clf_shortcuts, switch = setup_description()

    if isDeveloper:
        while True:
            print("Classifiers' definitions declared in 'models.py'")
            print(clf_str_available)
            command = input('Would you like to train the classifiers? (y/n)')
            if command in ["y", "n"]:
                break
        if command == "y":
            execute_ml_pipeline(True)

    print('INFOMAIR - Group 30 - 2021')

    while True:
        print(clf_str_available)
        command = input(clf_str_input)
        if command in clf_shortcuts:
            break

    while True:
        sentence = input("Give a sentence to classify (Use EXIT to exit): ")
        if sentence == "EXIT":
            break

        predict_with_bow(sentence.lower(), switch.get(command))


def setup_description():
    """
    Function that prepares all the statements to be shown throw the CLI and also some internal functionalities in order
    to be generic (no classifier count limit)

    :return: clf_str_available, clf_str_input: str - string to show through the console
                clf_shortcuts: list<str> - array containing the shortcuts for each classifier
                switch: map - dictionary that contains all the paths to each saved model
    """
    clf_names = get_classifier_names()

    clf_str_available = "Classifiers available: {} (base), {} (rule-based), ".format(clf_names[0], clf_names[1])
    clf_str_input = "Please choose a classifier: ('base', 'rule-based', "
    clf_shortcuts = ["base", "rule-based"]

    switch = {"base": "../models/ml0", "rule-based": "../models/ml1"}

    for i, name in enumerate(clf_names[2:]):
        clf_str_available += "{} (ml{}), ".format(name, i + 1)
        clf_str_input += "'ml{}', ".format(i + 1)
        clf_shortcuts.append("ml{}".format(i + 1))
        switch["ml{}".format(i + 1)] = "../models/ml{}".format(i + 1)

    clf_str_available = clf_str_available[:-2]
    clf_str_input = clf_str_input[:-2]
    clf_str_input += "): "

    return clf_str_available, clf_str_input, clf_shortcuts, switch


def predict_with_bow(sentence, model_path):
    """ Use a given model and bag of words to predict to which class the given sentence belongs.

    :param sentence: Sentence to classify
    :param model_path: Path to the joblib/H5 file containing the requested model.
    """
    if "joblib" in glob.glob(model_path + ".*")[0]:
        clf = load(model_path + ".joblib")
    else:
        clf = keras.models.load_model(model_path + ".h5")
    bow = load("../models/bow.joblib")
    print("Predicted class: {}".format("".join(clf.predict(bow.transform([sentence])))))


if __name__ == "__main__":
    main()
