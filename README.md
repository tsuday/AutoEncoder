# AutoEncoder
Simple implementation of AutoEncoder, which is one type of deep learning algorithm.
This is implemented based on Tensorflow.

## Reading data
This implementation requires file _dataList.csv_ , listing 2 file paths in each row.
CSV is necessary because batch reading data function of Tensorflow uses it.
This repository include sample csv and image files as samples.

## Requirements
Tensorflow 1.0, and libraries it requires.

Note: Currently, this implementation can treat only 1-dimensional image (e.g. gray scale image).
      I am planning to extend it to be able to treat 3-dimensional image (e.g. RGB color image).
