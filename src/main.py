from sklearn import tree

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
        word = input("Give a word to classify (Use EXIT to exit):")
        if word == "EXIT":
            break

        switch.get(command)(word)


def classify_base(word):
    print('chose base classifier')
    raise NotImplementedError()


def classify_rule_based(word):
    print('chose rule based classifier')
    raise NotImplementedError()


def classify_ml_1(word):
    print('chose ml1 classifier')
    raise NotImplementedError()


def classify_ml_2(word):
    print('chose ml2 classifier')
    raise NotImplementedError()


if __name__ == "__main__":
    main()
