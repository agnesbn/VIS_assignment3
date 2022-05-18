"""
Feature extraction with the CIFAR10 dataset
"""
""" Import the relevant packages """
 # system
import os
 # tf tools
import tensorflow as tf
 # argument parser
import argparse
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
from tensorflow.keras.optimizers import SGD
 # scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
 # plotting
import numpy as np
import matplotlib.pyplot as plt

""" Basic functions """
# Function to save history
def save_history(H, epochs, plot_name):
    outpath = os.path.join("out", f"{plot_name}.png")
    plt.style.use("seaborn-colorblind")
    
    plt.figure(figsize=(12,6))
    plt.suptitle(f"History for CIFAR_10 trained on VGG16", fontsize=16)
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="Validation", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="Validation", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(outpath))

# Function for saving classification report
def report_to_txt(report, report_name, epochs, learning_rate, batch_size):
    outpath = os.path.join("out", f"{report_name}.txt")
    with open(outpath,"w") as file:
        file.write(f"Classification report\nData: CIFAR_10\nModel: VGG16\nEpochs: {epochs}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\n")
        file.write(str(report))    

# Min-max normalisation function
def minmax(data):
    X_norm = (data-data.min())/(data.max()-data.min())
    return X_norm         
        
# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    # learning rate argument
    ap.add_argument("-l", 
                    "--learning_rate",
                    type=float,
                    default=0.001,
                    help="The learning rate for the stochastic gradient descent (default=0.01)")
    # batch size argument
    ap.add_argument("-b",
                    "--batch_size",
                    type=int,
                    default=128,
                    help="The size of the batches that the model goes through the data in (default=128)")
    # number of epochs to train the model in
    ap.add_argument("-e",
                    "--epochs",
                    type=int,
                    default=20,
                    help = "The number of epochs to train your model in (default=20)")
    # report name argument
    ap.add_argument("-r",
                    "--report_name",
                    type=str,
                    default="classification_report",
                    help="The name of the classification report")
    # plot name argument
    ap.add_argument("-p",
                    "--plot_name",
                    type=str,
                    default="history_plot",
                    help="The name of the plot of loss and accuracy")
    args = vars(ap.parse_args())
    return args 

""" Feature extraction with VGG16 """        
def train_model(learning_rate, batch_size, epochs, report_name, plot_name):
    # import cifar10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # simple normalisation
    X_train_scaled = minmax(X_train)
    X_test_scaled = minmax(X_test)
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
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = VGG16(include_top = False,
              pooling = "avg",
              input_shape = input_shape)
    # disable training convolutional layers
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = "relu")(flat1)
    output = Dense(10, activation = "softmax")(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # define gradient descent
    sgd = SGD(learning_rate=learning_rate)
    # compile model
    model.compile(optimizer=sgd,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    # train the model
    H = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size = batch_size,
                  epochs = epochs,
                  verbose = 1)
    # save history
    save_history(H, epochs, plot_name)
    # create and save classification report
    predictions = model.predict(X_test, batch_size=128)
    report = classification_report(y_test.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names = labels)
    report_to_txt(report, report_name, epochs, learning_rate, batch_size)
    return print(report)

def main():
    # parse arguments
    args = parse_args()
    # get arguments
    learning_rate = args["learning_rate"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    report_name = args["report_name"]
    plot_name = args["plot_name"]
    # train model, save plot and classification report
    train_model(learning_rate, batch_size, epochs, report_name, plot_name)


if __name__=="__main__":
    main()