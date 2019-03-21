# Mask-R-CNN-
Summary:
Using openCV DNN library to implement Mask R-CNN for object detection in C++
OS: Ubuntu 18.04.02 LTS
Complier: Cmake
---------------


What I learned:

-Theory behind Mask R-CNN: Instance Segmentation, R-CNN, Inception Model 

-Interfacing with Deep Neural networks library in OpenCV 4/2

-Working with frozen models and pretrained weights

-C++ implementation of OpenCV libraries


Mask R-CNN is the combination of using a R-CNN with instance segmentation to provide a mask which aims to capture the maximum number of pixels within an object detection bounding box. The backone of the R-CNN used was the Inception model

What is the Inception Model?
https://cloud.google.com/tpu/docs/inception-v3-advanced

A convoluted neural network which combines the use of dropout, average pooling, max pooling, and softmax. 
It provides the network with freedom to choose which combines provides the most accurate outcomes.  

What is a R-CNN?

Instead of searching throughout the entire image, for example  by using a sliding window methord, Regional - CNNs use a region proposal network to first predict where an object might be. This can decrease computatin because not all parts of the image is being processed.   

What is Region proposal Network?

Typically a triditional computer vision methord that identifies posible 


What is Instance Segmentation?


___
Aplication: 
What parameters did you change? 
Confidence in bounding box and confidence in mask.


