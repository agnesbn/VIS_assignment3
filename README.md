# Assignment 3 – Transfer learning + CNN classification
The portfolio for __Visual Analytics S22__ consists of 4 projects (3 class assignments and 1 self-assigned project). This is the __third assignment__ in the portfolio.

**Github repository**: https://github.com/agnesbn/VIS_assignment3.

## 1. Contribution
The initial assignment was made partly in collaboration with others from the course, but the final code is my own.

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


## 4. Usage

Before running the script, run the following in the Terminal:
```
pip install --upgrade pip
pip install scikit-learn tensorflow
sudo apt-get update
sudo apt-get -y install graphviz
```
Then, from the `VIS_assignment1` directory, run:
```
python src/image_search_hist.py --image_index {INDEX}
```
`{INDEX}` represents a user-defined argument. Here, you can write any number from 0–1359 and it will index your target image.

## 5. Discussion of results
