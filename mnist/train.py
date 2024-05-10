import argparse
import logging

import keras
import mlflow.keras
import mlflow.pyfunc
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
import onnx
import tf2onnx
import tensorflow as tf


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(
            {
                "loss": logs["loss"],
                "accuracy": logs["accuracy"],
                "val_loss": logs["val_loss"],
                "val_accuracy": logs["val_accuracy"],
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--epochs", "-e", type=int, default=4)

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    digits_labels = 10
    image_rows = 28
    image_cols = 28

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        train_data = train_data.reshape(
            train_data.shape[0], 1, image_rows, image_cols
        )
        test_data = test_data.reshape(
            test_data.shape[0], 1, image_rows, image_cols
        )
        input_shape = (1, image_rows, image_cols)
    else:
        train_data = train_data.reshape(
            train_data.shape[0], image_rows, image_cols, 1
        )
        test_data = test_data.reshape(
            test_data.shape[0], image_rows, image_cols, 1
        )
        input_shape = (image_rows, image_cols, 1)

    train_data = train_data.astype("float32")
    test_data = test_data.astype("float32")
    train_data /= 255
    test_data /= 255
    logging.info(f"Training data shape: {train_data.shape}")
    logging.info(f"{train_data.shape[0]} training samples")
    logging.info(f"{test_data.shape[0]} testing samples")

    train_labels = keras.utils.to_categorical(train_labels, digits_labels)
    test_labels = keras.utils.to_categorical(test_labels, digits_labels)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(digits_labels, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    model.fit(
        train_data,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(test_data, test_labels),
        callbacks=[CustomCallback()],
    )
    score = model.evaluate(test_data, test_labels, verbose=0)

    logging.info(f"Test loss: {score[0]}")
    logging.info(f"Test accuracy: {score[1]}")

    onnx_model_name = 'mnist.onnx'

    onnx_model = tf2onnx.convert.from_keras(model, test_data[0])
    onnx.save_model(onnx_model, onnx_model_name)

    mlflow.onnx.load_model(onnx_model_name)

    logging.info(mlflow.active_run().info.run_uuid)
