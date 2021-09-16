from joblib import load
from models import execute_ml_pipeline
import argparse

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

    if isDeveloper:
        while True:
            command = input('Would you like to train the classifiers? (y/n)')
            if command in ["y", "n"]:
                break
        if command == "y":
            execute_ml_pipeline(True)

    print('INFOMAIR - Group 30 - 2021')

    while True:
        command = input('Please choose a classifier: ("base", "rule-based", "ml1", "ml2")')
        if command in ["base", "rule-based", "ml1", "ml2"]:
            break

    switch = {
        "base": "../models/ml0.joblib",
        "rule-based": "../models/ml1.joblib",
        "ml1": "../models/ml2.joblib",
        "ml2": "../models/ml3.joblib"
    }

    while True:
        sentence = input("Give a word to classify (Use EXIT to exit):")
        if sentence == "EXIT":
            break

        predict_with_bow(sentence.lower(), switch.get(command))


def predict_with_bow(sentence, model_path):
    """ Use a given model and bag of words to predict to which class the given sentence belongs.

    :param sentence: Sentence to classify
    :param model_path: Path to the joblib file containing the requested model.
    """
    clf = load(model_path)
    bow = load("../models/bow.joblib")
    print("Predicted class: {}".format("".join(clf.predict(bow.transform([sentence])))))


if __name__ == "__main__":
    main()
