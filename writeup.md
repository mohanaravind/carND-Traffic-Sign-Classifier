#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image4]: ./examples/web/3.jpg "Traffic Sign 1"
[image5]: ./examples/web/14.jpg "Traffic Sign 2"
[image6]: ./examples/web/17.jpg "Traffic Sign 3"
[image7]: ./examples/web/22.jpg "Traffic Sign 4"
[image8]: ./examples/web/25.jpg "Traffic Sign 5"
[dataset]: ./assets/dataset.png "Data set"
[lenet]: ./assets/lenet5.png "LeNet"
[preprocess]: ./assets/preprocess.PNG "Preprocess"
[augment]: ./assets/augment.PNG "Augment"
[transrot]: ./assets/trans_rot.PNG "Translate and rotate"
[visual]: ./assets/visual.PNG "Visual"
[prediction]: ./assets/prediction.PNG "Prediction"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32 x 32**
* The number of unique classes/labels in the data set is **43**

####2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread between different labels we have. One thing that will be obvious looking at the chart is the dataset is biased towards certain labels. We don't have uniform data available for all the labels. This poses the risk of the network not performing well for all labels

![alt text][dataset]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

####1. Pre-processing

![alt text][preprocess]

###### Minimax
Used minimax algorithm to do the normalization of the images. This helps in keeping the mean and variance to the desired values

###### Grayscale
I experimented by testing the network with color and without color. There was no significant difference between them. This has also been highlighted in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann LeCun. This made me choose the decision to convert the inputs to grayscale by removing the color channels. This optimizes by reducing the amount of information that gets feed into our network without affecting the performance

###### Exposure Normalization
The image has different intensity based on the lighting condition. To normalize the exposure across all the images I used the skimage's equalize_adapthist to create an equal exposure on the images irrespective of the original exposure. This helps the network detect the curves and patterns better. Since I did not manage to create a vectorized implementation of this step I had to serialize the output to reduce the time it took to retrain after any variation to the inputs. This step usually takes the longest time in the preprocessing pipeline

##### Augmentation


If we look at the dataset we can clearly see that we lack variation and also the amount of data. For deep learning to be successful we need lots of data. It is practically very difficult for me to really travel to Germany and get more pictures of road signs and label it. So a better way was to augment the existing data and build new ones. 

###### Translation and rotation
I initially tested with translation and rotation of images where I managed to multiply the dataset by a factor of 14. By translating the sign within the image to the 4 corners of the image and rotating the image by +20 and -20 degree I got extra dataset. The dataset needed little more refinement in the way I was augmenting. I realized this only after noticing that my network was performing really poor. So I discarded this method. I know I can do better with this but decided it needs little more effort. One thing that could be changed so I could have managed with this is modifying the image back to the original size before doing translation or rotation and then clipping the image to the desired size. This is still a TODO

![alt text][transrot]

###### Zooming
Then I decided to do a simpler augmentation. Trying to zoom in the sign. I achieved this by using the coordinates of the sign that is available in the dataset and also using the original image size which is also available in the dataset. This easily doubled the number of examples in the dataset. I was noticing a significant difference in the accuracy. Initially without data augmentation I managed to get only 94% of accuracy and with this I managed to get close to 97%

![alt text][augment]


###### Note
While writing this I just realized I could have reduced the time it took for me to preprocess by doing the preprocessing first and then doing augmentation




####2. Final Model architecture

I tested with different architectures. Initially starting with the Lenet architecture and then experimented by adding extra inputs to the final fully connected layer as described in the paper referred above. Passing the convolution outputs from the initial layer while the dataset is not significantly large did not yield better result. So my final architecture became my initial architecture (Lenet)

![alt text][lenet]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image 						| 
| Convolution 5x5     	| 1x1 stride, valid padding, 6 filters      	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding      				|
| Convolution 5x5     	| 1x1 stride, valid padding, 16 filters      	|
| Fully connected		| This outputs to 120 units 					|
| Dropout       		| Keep prob of 0.6              				|
| Fully connected		| This outputs to 84  units 					|
| Dropout       		| Keep prob of 0.6              				|
| Fully connected		| This outputs to 43 units  					|
| Softmax				|   											|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used adam optimizer provided by tensorflow
Defined a class structure to play with the hyperparameters. This makes it convenient to easily pass around the data without having to add a new parameter to the methods.

###### Batch size
I tried out various batch sizes for the models. Ranging from 128, 256 and 512
Finally I settled with 512 since that was making my model run fast

###### Number of epochs
I experimented with this hyper parameter by trying out sometime 200, 100, 80, 50, 20, 10
Usually the model stops learning further due to lack of more variation in the dataset. So I finally ended up using 100 epochs to my model

###### Learning rate
The learning rates I tried out were 0.001 and 0.0001 for my model. I did not find the model to be stuck in any local optima so I went ahead with 0.001

###### Keep prob
This parameter is really useful to avoid over fitting. Regularization using dropout is preferred over other methods. I tried several values for this hyper parameter ranging from 0.3, 0.4, 0.5, 0.6, 0.7 and then settled with 0.6; One must note that I set the keep prob as 1.0 when we are predicting (actual run)


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
###### Training Accuracy = 100 %
###### Validation Accuracy = 97 %
###### Test Accuracy = 95 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* Initially I t
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][prediction]

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][visual]



##References
###Pre-Processing
* [http://sebastianraschka.com/Articles/2014_about_feature_scaling.html](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
* [https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale](https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale)
* [https://github.com/gwding/draw_convnet](https://github.com/gwding/draw_convnet)
* [http://john.freml.in/opencv-rotation](http://john.freml.in/opencv-rotation)
