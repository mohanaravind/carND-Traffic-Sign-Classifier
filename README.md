# **Traffic Sign Recognition** 

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
[dataset]: ./assets/dataset.PNG "Data set"
[lenet]: ./assets/lenet5.png "LeNet"
[preprocess]: ./assets/preprocess.PNG "Preprocess"
[augment]: ./assets/augment.PNG "Augment"
[transrot]: ./assets/trans_rot.PNG "Translate and rotate"
[visual]: ./assets/visual.PNG "Visual"
[prediction]: ./assets/prediction.PNG "Prediction"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

#### 1. This writeup includes all the rubric points and how I addressed each one. =
The link to the [project code](https://github.com/mohanaravind/carND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the Data Set

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32 x 32**
* The number of unique classes/labels in the data set is **43**

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread between different labels we have. One thing that will be obvious looking at the chart is the dataset is biased towards certain labels. We don't have uniform data available for all the labels. This poses the risk of the network not performing well for all labels

![alt text][dataset]

### Design and Architecture


#### 1. Pre-processing

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


#### 2. Final Model architecture

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
 

#### 3. Hyperparameters

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


#### 4.The Solution

My final model results were:
###### Training Accuracy = 100 %
###### Validation Accuracy = 97 %
###### Test Accuracy = 95 %

If an iterative approach was chosen:
* Initially I took the approach of using LeNet architecture. This is a simple architecture and has performed well with ImageNet dataset. 
* The network was overfitting the data. This was observed when validation accuracy was low but training accuracy was high. 
* To avoid overfitting I added a dropout layer between the two fully connected layers
* Modifying the learning rate, batch size, epochs and dropout rate were some of the things I did to tune the network
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* A convolution layer reduces the number of weights one needs to use to train the model. Having a regular fully connected neural network would mean we will have enormous amount of weights that needs to be adjusted
* Pooling layers helps in down sampling the inputs and thereby makes it easy to manage the training
* Traffic sign detection is an image detection problem. LeNet seems to be a good candidate to solve it

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images from the original website were of different sizes. I had to scale them to fit to the size the model accepts (32 x 32).
The orientation of each of these images are different. Some are tilted or little skewed and that could make it difficult for the network to detect the right sign

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)      		|  Traffic signals  									| 
| Stop Sign      		| Stop sign   									| 
| No entry     			| No entry								|
| Bumpy road					| Bumpy road											|
| Road work      		| Bicycles crossing 				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

![alt text][prediction]

The model is performing really well on the cases where we have good number of samples available. If we look back at the sample distrubtion bar chart we can clearly see that Road work, Bumpy road signs had really low samples. Stop sign had one of the highest samples and we could clearly see a correlation with the sample size and the prediction confidence

### Visualizing the Neural Network

#### Visualizing the First convolution layer output when network is detecting a no entry sign
This shows how the initial layer of a convolution layer tries to detect the edges and other primitive feature. We can see the weights/features corresponding to this filter is firing up and we can now visualize them over here

![alt text][visual]



## References
* [http://sebastianraschka.com/Articles/2014_about_feature_scaling.html](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
* [https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale](https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale)
* [https://github.com/gwding/draw_convnet](https://github.com/gwding/draw_convnet)
* [http://john.freml.in/opencv-rotation](http://john.freml.in/opencv-rotation)
