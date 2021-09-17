from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# Doesn't work with GridSearchCV
# class BasicKerasClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, input_dimension):
#         self.model = get_model(input_dimension)
#         pass
#
#     def fit(self, x, y=None):
#         return self.model.fit(x, y)
#
#     def predict(self, x, y=None):
#         return np.argmax(self.model.predict(x), axis=-1)


def setup_model(input_dimension):  # Number of features
    """
    Returns a compiled model with the characteristics defined in the function

    :param input_dimension: int - the input dimension for the first NN-layer
    :return: model: Sequential - the compiled model
    """
    # Define
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dimension, activation="relu"))
    model.add(layers.Dense(15, activation="softmax"))
    # Compile
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])
    return model


def get_model(input_dimension):
    """
    Creates a scikit-learn compatible Keras model

    :param input_dimension: int - the input dimension for the first NN-layer
    :return: KerasClassifier - the classifier
    """
    return KerasClassifier(build_fn=lambda: setup_model(input_dimension), verbose=0)
