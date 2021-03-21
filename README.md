# Bird-Species-Classifier

Deep learning model to classify input of image of a bird to the respective species., and return top 3 predictions for the input image

# Introduction

The model is trained on a dataset which has 250 different classes, hence can classify into 250 different species.
On running ```main.py```, we can input an image of a bird, and the model will classify it into it's respective species.

# Dataset

The dataset is derived from Kaggle, and is linked below.

* It is structured to have *training*, *test* and *validation* folders.
* It consists of 250 distinct classes, with images in all three folders.
* It also includes a consolidation folder, which consists of all images from all the sub-folders.

Dataset Link: https://www.kaggle.com/gpiosenka/100-bird-species

Data has been scaled down to 224 x 224, in order to have uniformity and ease while training the model.

# Model Features

> Resnet 152
 
 * ResNet152V2 is the pretrained model that is used for this particular classification.
 * Model used is trained with the weights of ```imagenet``` dataset.
 * Using 3 color channels 'rgb'

> Artificial Deep Neural Network

  * Consists of GlobalAveragePooling layer to reduce dimensionality and to flatten the output data from feature extraction layers.
  * Usage of ```Dropout``` to prevent overfitting and improving accuracy.
  * Usage of ```BatchNormalization``` to stabilize learning process and *reduce* number of epochs required for higher accuracy.
  * Output layer has 250 neurons, for classification into 250 classes. 
  * softmax is used on the last layer to return it as probability density, for easier classification.
  * ReLu is used as it is effiient and easy to train.
  
# Compiling and Training Model

  * Model is compiled with loss function as ```categorical_crossentropy```
  * Optimizer used is ```adam```. learning rate is modified to 1e-04.
  * Model is trained for 50 epochs for a decent accuracy.

# Accuracy and Overfitting

* Dropout layer is used generously to assure high performance
* BatchNormalization to reduce overfitting
* Data Augmentation is used to increase size of dataset, to obtain a more comprehensive data, to us layer.

# Accuracy Results

   At the last epoch:
  
   Training Accuracy | Validation Accuracy
   ------------------|---------------------
   88.76%            | 94.48%

# Graphs

* Usage of ```matplotliob.pyplot```
* Graph that plots **Accuracy vs Epochs**

Accuracy Graph: 

![Unknown](https://user-images.githubusercontent.com/73730958/111886207-7e5c4800-89f2-11eb-95ec-eacafc8cad71.png)

* Graph that plots **Loss vs Epochs**

Loss Graph: 

![Unknown](https://user-images.githubusercontent.com/73730958/111886239-aa77c900-89f2-11eb-8db7-d0117711f038.png)

* As can be made out from the graph, the model is not overfitting, and has a great accuracy.

# Requirements

***Libraries required are:***

```
tensorflow
numpy
PIL
```

# Usage

* Run ```main.py``` from the command line.
* Input the path of the image when prompted.
* The output will display the input image as well as the top 3 predictions with their respective probabilities.

# Sample Execution

![scs](https://user-images.githubusercontent.com/73730958/111894084-df554180-8a2d-11eb-9dc0-61e4e530af11.jpg)

The model accurately predicts with high confidence(99.97%) that the input image is a bald eagle, and the other two top probabilities are negligible.

# Presentation Link

https://drive.google.com/file/d/10cFkQABKTnY8CMduLNN8nJuAoHnW0OgC/view?usp=sharing

Team:
* Prateek Rao
* Suraj Rao KP
* Sundeep A

