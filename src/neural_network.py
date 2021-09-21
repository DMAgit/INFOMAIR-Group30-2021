from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from src.classifier import Classifier


class BasicKerasClassifier(BaseEstimator, ClassifierMixin, Classifier):
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.model = KerasClassifier(build_fn=lambda: self.setup_model(), verbose=0)
        pass

    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return np.argmax(self.model.predict(x), axis=-1)

    def save_to_file(self):
        self.model.save(self.get_file_name())

    def get_file_name(self):
        return "../models/neural-network.h5"

    def load_from_file(self):
        keras.models.load_model(self.get_file_name())

    def get_name(self):
        return "neural-network"

    def setup_model(self):  # Number of features
        """
        Returns a compiled model with the characteristics defined in the function

        :param input_dimension: int - the input dimension for the first NN-layer
        :return: model: Sequential - the compiled model
        """
        # Define
        model = Sequential()
        model.add(layers.Dense(10, input_dim=self.input_dimension, activation="relu"))
        model.add(layers.Dense(15, activation="softmax"))
        # Compile
        model.compile(loss=SparseCategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])
        return model

    def set_grid_search_cv(self, params, x_train, y_train):
        grid = GridSearchCV(estimator=self.model, param_grid=params, cv=10, n_jobs=-1,
                            scoring="f1_macro")
        grid.fit(x_train, y_train)
        self.model = grid.best_estimator_
        return self
