from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.base import BaseEstimator, ClassifierMixin
from src.classifier import Classifier
from tensorflow.keras.optimizers import Adam
import numpy as np


class BasicKerasClassifier(BaseEstimator, ClassifierMixin, Classifier):
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.model = KerasClassifier(build_fn=self.setup_model, verbose=1)
        pass

    def fit(self, x, y=None):
        self.model.fit(x.sorted_indices(), y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def save_to_file(self):
        self.model.model.save(self.get_file_name())

    def get_file_name(self):
        return "../models/neural-network.h5"

    def load_from_file(self):
        self.model = keras.models.load_model(self.get_file_name())
        return self

    def get_name(self):
        return "neural-network"

    def setup_model(self, lr=0.01, dropout=0.0):  # Number of features
        """
        Returns a compiled model with the characteristics defined in the function

        :param lr: float - TODO
        :param dropout: float - TODO
        :return: model: Sequential - the compiled model
        """
        # Define
        model = Sequential()
        model.add(layers.Dense(self.input_dimension, input_dim=self.input_dimension, activation="relu"))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(15, activation="softmax"))
        # Compile
        model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=lr),
                      metrics=["accuracy"])
        return model

    def set_grid_search_cv(self, params, x_train, y_train):
        grid = RandomizedSearchCV(estimator=self.model, param_distributions=params, cv=10, n_jobs=-1,
                                  scoring="f1_macro")
        grid.fit(x_train.sorted_indices(), y_train)
        self.model = grid.best_estimator_
        return self
