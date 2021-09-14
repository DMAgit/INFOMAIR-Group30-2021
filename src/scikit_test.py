import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# Load dataset
raw_data = dict()
with open("../data/dialog_acts.dat", "r") as f:
    for line in f:
        lines = line.split(" ")
        key = lines[0]
        value = " ".join(lines[1:]).strip("\n")
        if key not in raw_data:
            raw_data[key] = [value]
        else:
            if value not in raw_data[key]:
                raw_data[key].append(value)

# print(raw_data)

# Prepare X and y variables
X = []
y = []
for key in raw_data:
    for statement in raw_data[key]:
        X.append(statement)
        y.append(key)

print(X)
print(y)

# One Hot Encoding
df_y = pd.get_dummies(y)

# Split train/test 85/15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

# Transform to Bag of Words
bow = CountVectorizer()
X_train = bow.fit_transform(X_train)

# Baseline 1 (majority class -> inform)
clf = DummyClassifier(strategy="most_frequent")
clf.fit(X_train, y_train)

# Baseline 2 (rule-based)


# CLF 1


# CLF 2


# Prediction
y_pred = clf.predict(bow.transform(X_test))

# Results' report
print(classification_report(y_test, y_pred))
