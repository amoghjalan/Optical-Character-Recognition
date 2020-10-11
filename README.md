# Optical-Character-Recognition

This project is a Classification model which classifies the images for Handwritten digits.

## About Dataset

The dataset used for trang the model is MNIST Handwritten Digits Dataset.

## Running the model -1

Inorder to evaluate the model on your own dataset:

- Make your dataset in the same format as of MNIST training dataset.
- Then run the ocr.sh file
- When it asks for path of dataset, enter the full path to it.
  Example:

```
$ ./ocr.sh
Enter path to the dataset:[YOUR FILE PATH]
```

## Running the Model -2

If your dataset is a collection of images then:

- Run the ocr1.sh file
- When it asks for path of dataset, enter the full path to it.
  Example:

```
$ ./ocr.sh
Enter path to the dataset:[YOUR FILE PATH]
```

- It will make a file named output.txt with all the predicted labels.
