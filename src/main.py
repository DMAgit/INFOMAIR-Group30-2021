import json

from joblib import load
from models import execute_ml_pipeline, get_classifier_names, get_classifiers
import argparse

from src.state_manager import initialize_state, update_state

""" Entry point to the application
"""

default_settings_path = '../settings.json'


def main():
    """
    Entry point of application, writes instructions and takes input.
    """
    # Add parser for flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--settings')

    is_developer = parser.parse_args().development
    settings_path = parser.parse_args().settings

    clf_str_available, clf_str_input, clf_shortcuts, switch = setup_description()

    if is_developer:
        while True:
            print("Classifiers' definitions declared in 'models.py'")
            print(clf_str_available)
            command = input('Would you like to train the classifiers? (y/n): ')
            if command in ["y", "n"]:
                break
        if command == "y":
            execute_ml_pipeline(True)

    print('INFOMAIR - Group 30 - 2021')

    if settings_path is None:
        settings_path = input('Give the path to the settings file (leave empty to use default repository file.)')
        if settings_path == '':
            settings_path = default_settings_path

    settings = get_settings(settings_path)

    if settings['chooseClassifier']:
        while True:
            print(clf_str_available)
            command = input(clf_str_input)
            if command in clf_shortcuts:
                break
    else:
        command = 'sgd'

    while True:
        state = initialize_state(switch.get(command))
        while state.state_number < 8:
            sentence = input(state.get_question())
            update_state(state, switch.get(command), sentence)
        print(state.get_question())
        break

        # Old
        # sentence = input("Give a sentence to classify (Use EXIT to exit): ")
        # if sentence == "EXIT":
        #     break
        #
        # predict_with_bow(sentence.lower(), switch.get(command))


def setup_description():
    """
    Function that prepares all the statements to be shown throw the CLI and also some internal functionalities in order
    to be generic (no classifier count limit)

    :return: clf_str_available, clf_str_input: str - string to show through the console
                clf_shortcuts: list<str> - array containing the shortcuts for each classifier
                switch: map - dictionary that contains all the paths to each saved model
    """
    classifiers = get_classifiers()
    clf_names = get_classifier_names()

    combined_names = "', '".join(clf_names)
    combined_names = f"'{combined_names}'"
    clf_str_available = f"Classifiers available: {combined_names}"
    clf_str_input = f"Please choose a classifier: ({combined_names}): "
    clf_shortcuts = clf_names

    switch = {}

    for i, classifier in enumerate(classifiers):
        switch[classifier[0].get_name()] = classifier[0]

    return clf_str_available, clf_str_input, clf_shortcuts, switch


def predict_with_bow(sentence, classifier):
    """ Use a given model and bag of words to predict to which class the given sentence belongs.

    :param sentence: Sentence to classify
    :param classifier: Classifier to use
    """

    clf = classifier.load_from_file()

    bow = load("../models/bow.joblib")

    prediction = clf.transform_and_predict(sentence, bow)

    print(f"Predicted class: {prediction}")


def get_settings(path):
    """
    Read the settings file and parse it as an object
    :param path: Path to the settings json
    :return: The settings as an object
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print("can't find given file, using default settings instead")
        try:
            with open(default_settings_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print("can't find default settings either, exiting")
            exit()


if __name__ == "__main__":
    main()
