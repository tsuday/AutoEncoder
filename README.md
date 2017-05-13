# AutoEncoder
Simple implementation of AutoEncoder, which is one of the deep learning algorithms.
This is implemented based on framework Tensorflow.

## Learning data
This implementation requires file _dataList.csv_ , listing 2 file paths in each row as learning data.
File of the first column path is input for AutoEncoder, and the other one of the second column path is 
CSV is necessary because batch data read function of Tensorflow needs it.
This repository include sample csv and image files as samples.

_Currently, data images have to be 512x512 pixels size._

## How to execute
Please simply run _AutoEncoder.py_.<br>
By default implementation and settings, sample data is used for learning.<br>
To run, libraries as stated below in _Requirements_ is necessary.

## Explanations of Source Code

### Overview
_AutoEncoder.py_ consists of two parts.

* Definition part for class _AutoEncoder_
* Runnning part of _AutoEncoder_ for learning

### Customize and Settings
_TODO:to be written_

### Output data while learning
While learning, the implementation output the status of learning as below.

#### Standart output
Running part will write learning progress like below.<br>
_Step: 600, Loss: 90405240.000000 @ 2017/05/12 08:50:08_

This consists of 3parts:<br>
* Step value of learning process
* Loss value, which is sequare sum of pixel value differences between predicted values and teacher values
* Timestamp when the progress is output
<br>
Default implementation of running parts write progress at the first step and every 200 steps after the first step.

#### Images predicted
_TODO:to be written_

#### Tensor Board
_TODO:to be written_

## Session file to resume learning
_TODO:to be written_

## Prediction
_TODO:to be written_

## Requirements
Python3, Tensorflow 1.1.0, and the libraries it requires.

Note: Currently, this implementation can treat only 1-dimensional image (e.g. gray scale image).
      I am planning to extend it to be able to treat 3-dimensional image (e.g. RGB color image).
