import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    print('Hello World')

    f = open("../data/dialog_acts.dat", "r")
    data = list(map(lambda x: x.split(" ", 1), f.readlines()))
    dataset = tf.data.Dataset.from_tensor_slices(data)

    list(dataset.as_numpy_iterator())

    model = keras.Sequential(
        [
            layers.Dense(1, activation="relu", name="layer1")
        ])

    #model.fit(dataset)


if __name__ == "__main__":
    main()
