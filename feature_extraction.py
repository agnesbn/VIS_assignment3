"""
Feature extraction with the CIFAR10 dataset
"""
# Import the necessary packages
 # system
import os
 # tf tools
import tensorflow as tf
 # image processsing
from tensorflow.keras.preprocessing.image import (load_img, img_to_array, ImageDataGenerator)
 # VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input, decode_predictions,VGG16)
 # cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10
 # layers
from tensorflow.keras.layers import (Flatten, Dense, Dropout, BatchNormalization)
 # generic model object
from tensorflow.keras.models import Model
 # optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
 # scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
 # plotting
import numpy as np
import matplotlib.pyplot as plt

""" Basic functions """
# Function for saving loss and accuracy after every epoch
def save_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join("out", "history.png"))

# Function for saving classification report
def report_to_txt(report):
    outpath = os.path.join("out", "classification_report.txt")
    with open(outpath,"w") as file:
        file.write(str(report))    

""" Feature extraction with VGG16 """        
def train_model():
    # import cifar10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # simple normalisation
    X_train_scaled = X_train/255
    X_test_scaled = X_test/255
    # import labels
    labels = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]
    # binarise the labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    # clear the layers kept in memory (in case the code is run repeatedly)
    tf.keras.backend.clear_session()
    # load VGG16 without classifier layer
    model = VGG16(include_top = False,
              pooling = "avg",
              input_shape = (32,32,3))
    # disable training convolutional layers
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = "relu")(flat1)
    output = Dense(10, activation = "softmax")(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile the model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.01,
        decay_steps = 10000,
        decay_rate = 0.9)
    # define gradient descent
    sgd = SGD(learning_rate=lr_schedule)
    # compile model
    model.compile(optimizer=sgd,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    # train the model
    H = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size = 128,
                  epochs = 10,
                  verbose = 1)
    # save history
    save_history(H, 10)
    # create and save classification report
    predictions = model.predict(X_test, batch_size=128)
    report = classification_report(y_test.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names = labels)
    report_to_txt(report)

def main():
    train_model()
   
    
if __name__=="__main__":
    main()