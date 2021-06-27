# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[visual]: ./report_image/visual_histogram.png "Visualization"
[grayscale]: ./report_image/grayscale.jpg "Grayscaling"
[aug_image]: ./report_image/aug_image.png "Random Noise"
[traffic_sign1]: ./report_image/test_image1.png "Traffic Sign 1"
[traffic_sign2]: ./report_image/test_image2.png "Traffic Sign 2"
[traffic_sign3]: ./report_image/test_image3.png "Traffic Sign 3"
[traffic_sign4]: ./report_image/test_image4.png "Traffic Sign 4"
[traffic_sign5]: ./report_image/test_image5.png "Traffic Sign 5"
[traffic_sign6]: ./report_image/test_image6.png "Traffic Sign 6"
[traffic_sign7]: ./report_image/test_image7.png "Traffic Sign 7"
[traffic_sign8]: ./report_image/test_image8.png "Traffic Sign 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/chifai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing all the training and testing data number which is quite unevenly distributed.

![alt text][visual]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is less important to classify a traffic sign. Converting images to grayscale should reduce the data size, thus increase the efficiency of the training process.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]

As a last step, I normalized the image data to make sure the data has zero mean and equal variance.

I decided to generate additional data because the number of test images from each class are very uneven as seen as above histogram.

For each class, if the number of training data is less than 1000, an image is randomly picked and is processed to be a new augmented image with one of the following: adjust saturation, adjust brightness, rotating the image, or scaling the image. The procedure is repeated until each class has at least 1000 images.

Here is the comparsion between few original images and augmented images:

![alt text][aug_image]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image						| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Convolution 3x3	    | input: 30x30x32, output: 28x28x32				|
| RELU					|												|
| Max pooling			| input: 28x28x32, output: 14x14x32         	|
| Flatten               | input: 14x14x32, output: 6272x1               |
| Fully connected		| input: 6272, output: 4096                  	|
| Sigmoid Activation	|               								|
| Fully connected		| input: 4096, output: 1024                  	|
| Sigmoid Activation	|               								|
| Fully connected		| input: 1024, output: 43                     	|
| Sigmoid Activation	|               								|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with learning rate 0.001. Every epoch is fed with a batch of 128 input images.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 0.999
* validation set accuracy of 0.961
* test set accuracy of 0.952

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * Six layers of Lenet. More layers are designed to extract more feature
* What were some problems with the initial architecture?
  * First few epochs have very low accuracy like less than 0.2.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * Adding more layers, making sure the map size big enough and trying 50+ epoch times yields better result to about merely more than 0.7.
* Which parameters were tuned? How were they adjusted and why?
  * More layers and filters are added to extract more features as possible
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? 
  * The most significant impact on the increase in accuracy is adding more augmented images. Without any additional augmented images, no matter how the architecture parameters is tuned and how many epoch times is executed, the highest accuracy is limited by ~0.85. After adding augmented images as above stated, accuracies can finally go up to >0.9.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs. Trying to be realistic, 7 of them are extracted on Google Map, meanwhile 2 of them are from official website as control group.

![alt text][traffic_sign1] ![alt text][traffic_sign2] ![alt text][traffic_sign3] 
![alt text][traffic_sign4] ![alt text][traffic_sign5] ![alt text][traffic_sign6]
 ![alt text][traffic_sign7] ![alt text][traffic_sign8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		    | Yield   							        	| 
| No Passing     		| Yield 										|
| Turn right ahead		| Turn right ahead								|
| Go straight or right	| Go straight or right							|
| Speed limit(50km/h)	| Speed limit(80km/h)							|
| Go straight or right	| Go straight or right							|
| Stop					| Stop											|
| Yield					| Yield											|


Except the second and fifth image, others are correct. I believe the second image is too small, whose surrounding has confused the algorithm a lot. While the fifth image is actually a 50km/h speed limit sign but is mistaken as 80km/h. I think it is kind of reasonable because 80 and 50 look quite similar.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Although 6 out of 8 guesses are correct, the top probability is quite small (<0.06)
I am not sure about the reason if the image extracted from Google Map is way too different from the samples.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.041                 | Yield   							        	| 
| 0.061                 | Yield 										|
| 0.061                 | Turn right ahead								|
| 0.061                 | Go straight or right							|
| 0.057                 | Speed limit(80km/h)							|
| 0.025                 | Go straight or right							|
| 0.061                 | Stop											|
| 0.061                 | Yield											|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


