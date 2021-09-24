from joblib import load
from models import execute_ml_pipeline, get_classifier_names, get_classifiers
import argparse

from src.state_manager import initialize_state, update_state

""" Entry point to the application
"""


def main():
    """
    Entry point of application, writes instructions and takes input.
    """
    # Add parser for flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', action='store_true')

    is_developer = parser.parse_args().development

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

    while True:
        print(clf_str_available)
        command = input(clf_str_input)
        if command in clf_shortcuts:
            break

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


if __name__ == "__main__":
    main()
