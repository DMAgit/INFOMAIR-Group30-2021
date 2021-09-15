from joblib import load

""" Entry point to the application
"""


def main():
    print('INFOMAIR - Group 30 - 2021')

    command = ''

    while True:
        command = input('Please choose a classifier: ("base", "rule-based", "ml1", "ml2")')
        if command in ["base", "rule-based", "ml1", "ml2"]:
            break

    switch = {
        "base": classify_base,
        "rule-based": classify_rule_based,
        "ml1": classify_ml_1,
        "ml2": classify_ml_2
    }

    while True:
        sentence = input("Give a word to classify (Use EXIT to exit):")
        if sentence == "EXIT":
            break

        switch.get(command)(sentence.lower())


def classify_base(sentence):
    clf = load("../models/ml0.joblib")
    bow = load("../models/bow.joblib")
    print("Predicted class: {}".format("".join(clf.predict(bow.transform([sentence])))))


def classify_rule_based(sentence):
    clf = load("../models/ml1.joblib")
    bow = load("../models/bow.joblib")
    print("Predicted class: {}".format("".join(clf.predict(bow.transform([sentence])))))


def classify_ml_1(sentence):
    clf = load("../models/ml2.joblib")
    bow = load("../models/bow.joblib")
    print("Predicted class: {}".format("".join(clf.predict(bow.transform([sentence])))))


def classify_ml_2(sentence):
    clf = load("../models/ml3.joblib")
    bow = load("../models/bow.joblib")
    print("Predicted class: {}".format("".join(clf.predict(bow.transform([sentence])))))


if __name__ == "__main__":
    main()
