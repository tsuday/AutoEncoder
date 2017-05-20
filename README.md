# AutoEncoder
Simple implementation of AutoEncoder, which is one of the deep learning algorithms.  
This is implemented based on framework Tensorflow.

## Overview
This project consists of three programs.

* _AutoEncoder.py_  
    Main program of this project, and this is the imeplementation of AutoEncoder. 
* _Trainer.py_  
    Sample program to run training. It requires training data images, and will be described below for details.
* _Predictor.py_  
    Sample program to prediction by using trained data. It requires trained parameter data and input for prediction. This will also be described below.

## AutoEncoder.py
Python class which has training and prediction functions of AutoEncoder.  
This class is implemented based on framework Tensorflow.

_TODO:to be written about more technical details._

### Learning data
This implementation requires CSV file listing 2 image file paths in each row as learning data.
File of the first column path is input for AutoEncoder, and the second column is file path of training data image.
CSV is necessary because batch data read function of Tensorflow needs it.
This repository include sample csv, _dataList.csv_, and image files as samples.
  
### Limitations
Currently, this implementation has some limitations.
  
* Treat only 1-dimensional image (e.g. gray scale image).
  There is a plan to extend it to treat 3-dimensional image (e.g. RGB color image).
* Data images have to be 512x512 pixels size._

### Customize and Settings
_TODO:to be written_

### Output data while learning
While learning, the implementation output the status of learning as below.

#### Standart output
Running part will write learning progress like below.  
_Step: 600, Loss: 90405240.000000 @ 2017/05/12 08:50:08_

This consists of 3parts:  
* Step value of learning process
* Loss value, which is sequare sum of pixel value differences between predicted values and teacher values
* Timestamp when the progress is output  
  
Default implementation of running parts write progress at the first step and every 200 steps after the first step.

#### Images predicted
_TODO:to be written_

#### Tensor Board
_TODO:to be written_

#### Session file to resume learning
_TODO:to be written_

## Trainer.py
Sample implementation of training by using AutoEncoder.py.
This implementation use data from  _dataList.csv_.  

_TODO:to be written about more technical details._

## Predictor.py
Sample implementation of prediction by using AutoEncoder.py.
This implementation use data from  _predictList.csv_.  

_TODO:to be written about more technical details._


## How to execute
1. Please run _AutoEncoder.py_, and class definition is loaded.<br>
2. Please run _Trainer.py_. This program load training data and output session data of Tensorflow for every 200 training steps.
3. Please modify settings in _Predictor.py_, and run. This program load session data and output predicted image by matplotlib.

To run, libraries as stated below in _Requirements_ is necessary.

## Requirements
* Python3, Tensorflow 1.1.0, and the libraries it requires.
* _Trainer.py_ and _Predictor.py_ require _matplotlib_ to draw predicted images.
