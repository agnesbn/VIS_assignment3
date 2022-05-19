# Assignment 3 – Transfer learning + CNN classification
The portfolio for __Visual Analytics S22__ consists of 4 projects (3 class assignments and 1 self-assigned project). This is the __third assignment__ in the portfolio.

## 1. Contribution
I wrote this code independently both at the initial hand-in date and for the final portfolio. The [`save_history()`](https://github.com/agnesbn/VIS_assignment3/blob/03a2fdd27c9c3faecbc9b0807a8b45425d893de4/src/feature_extraction.py#L32) function was  inspired by one provided to us by Ross during the course.

## 2. Assignment description
### Main task
In this assignment, you are still going to work with the CIFAR10 dataset. However, this time, you are going to make build a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction. 

Your ```.py``` script should minimally do the following:

- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report

### Bonus tasks
- Use ```argparse()``` to allow users to define specific hyperparameters in the script.
  - This might include e.g. learning rate, batch size, etc
- The user should be able to define the names of the output plot and output classification report from the command line

## 3. Methods
The [feature_extraction.py](https://github.com/agnesbn/VIS_assignment3/blob/main/src/feature_extraction.py) script loads the CIFAR_10 data set, performs a few processing functions on the data (reshaping and label binarisation), and then trains only the classifier layers of a pretrained model, VGG16. The final results are saved in the form of a classification report and a history plot.

The code allows for the user to define both some hyperparameters (learning rate, batch size, number of epochs) as well as the names of the output plot and classification report. If these are not provided, default values have been set for all.

## 4. Usage
### Install packages
Before running the script, run the following in the Terminal:
```
pip install --upgrade pip
pip install scikit-learn tensorflow
sudo apt-get update
sudo apt-get -y install graphviz
```

### Run the script
Make sure your current directory is the `VIS_assignment3` folder. Then run:
```
python src/feature_extraction.py (--learning_rate <LEARNING RATE> --batch_size <BATCH SIZE> --epochs <EPOCHS> --report_name <REPORT NAME> --plot_name <PLOT NAME>)
```
__Input:__

- `<LEARNING RATE>` represents the learning rate for the stochastic gradient descent. The default value is `0.001`.
- `<BATCH SIZE>` represents the batch_size by with the model goes through the data. The default valies is `128`.
- `<EPOCHS>` represents the number of epochs that the model trains in. The default value is `20`.
- `<REPORT NAME>` represents the name that the classification report is saved under (it will always be saved as a TXT). The default value is `classification_report`.
- `<PLOT NAME>` represents the name that the history plot is saved under (it will always be saved as a PNG). The default value is `history_plot`.


The results are saved in the [`out`](https://github.com/agnesbn/VIS_assignment3/tree/main/out) folder.

## 5. Discussion of results
After training the model with the default hyperparameters (20 epochs, a learning rate of 0.001, and a batch size of 128), an **accuracy of 55%** was reached. The training curves as seen in the history plot shows slightly conflicting tendencies. On the one hand, the loss curve seems to have flattened out and would most likely not improve much with more training time. The accuracy curve, on the other hand, seems to still be in development and might improve with more training. Changing the hyperparameters (epochs, learning rate and batch size) might improve on the results seen here.

![](https://github.com/agnesbn/VIS_assignment3/blob/main/out/history_plot.png)
