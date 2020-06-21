import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])


    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    print(len(x_test), len(x_train), len(y_test), len(y_train))
    x_train = x_train/255.0
    x_test = x_test/255.0
    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    label = 0
    finalImg = []
    finalLabel = []
    if os.path.exists(data_dir):
        listDir = os.listdir(data_dir)
        for dir_ in listDir:
            listImg = os.listdir(data_dir + "\\" + dir_)
            for img in listImg:
                if img[-3:] != "csv":
                    imgArray = cv2.imread(data_dir + "\\" + dir_ + "\\" + img)
                    imgArray = cv2.resize(imgArray, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
                    finalImg.append(imgArray)
                    finalLabel.append(label)
            label+=1
        return (finalImg, finalLabel)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(30, 30, 3)),
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CATEGORIES),
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()





