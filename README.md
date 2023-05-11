# PCA-MachineLearning

In this code, I will be using PCA to classify different classes in the MNIST dataset.

### Dataset:

I used and downloaded the widely available and popular **MNIST dataset** from the Pytorch library and created the data loaders accordingly. The training dataset contains 60,000 images while the testing dataset contains 10,000 images. The pixels of images are dimensions 28 by 28.

### Algorithm:

-   Compute the mean of train data.

-   Mean subtract the data.

-   Find the covariance matrix.

-   Compute eigenvalues and eigenvectors of the covariance matrix.

-   Compute the principal components and find how much information is
    captured by each principal component.

### Working:

After obtaining the principal components, the code randomly selects a data point from the test dataset and classifies it into one of the classes using the nearest neighbor in a low-dimensional space. It then reports the following performance metrics for each class and for all classes combined:

-   Precision

-   Recall

-   Accuracy

-   F1 score

-   Confusion matrix

-   ROC Curve

-   Area-Under-the-Curve

This is the core implementation of PCA with all its logic in PyTorch.
