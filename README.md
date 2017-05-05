# AutoEncoder
Simple implementation of AutoEncoder, which is one of the deep learning algorithms.
This is implemented based on framework Tensorflow.

## Learning data
This implementation requires file _dataList.csv_ , listing 2 file paths in each row as learning data.
File of the first column path is input for AutoEncoder, and the other one of the second column path is 
CSV is necessary because batch data read function of Tensorflow needs it.
This repository include sample csv and image files as samples.


## How to execute
_TODO:to be written_

### Output data while learning
While learning, the implementation output the status of learning as below.

#### Standart output
_TODO:to be written_

#### Images predicted
_TODO:to be written_

#### Tensor Board
_TODO:to be written_

## Session file to resume learning
_TODO:to be written_

## Prediction
_TODO:to be written_

## Requirements
Tensorflow 1.1.0, and the libraries it requires.

Note: Currently, this implementation can treat only 1-dimensional image (e.g. gray scale image).
      I am planning to extend it to be able to treat 3-dimensional image (e.g. RGB color image).
