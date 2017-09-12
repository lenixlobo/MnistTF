import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import time
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import math 

#print(tf.__version__)->1.0.0

#CONFIGURATION OF THE NEURAL NETWORK
#Convolutiona layer 1
filter_size1 = 5
num_filters = 16
#convolutional layer 2
filter_size2 = 5
num_filters2 = 36
#Fully connected layer
fc_size = 128	#number of neurons in fully connected layer


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST',one_hot=True)

print("size of:")
print("-Training-set :\t\t{}").format(len(data.train.labels))
print("-Test-set:\t\t{}".format(len(data.test.labels)))
print("-Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels,axis=1)

#Define the data dimensions

#we have 28 pixels in each dimension in MNIST dataset
img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size,img_size)
num_channels = 1
num_classes = 10#one class for each of 10 digits

def plot_images(images,cls_true,cls_pred=None):
	assert len(images)==len(cls_true)==9

	#Create figure with 3X3 sub-plots
	fig,axes = plt.subplots(3,3)
	fig.subplots_adjust(hspace=0.3,wspace=0.3)

	for i,ax in enumerate(axes.flat):
		#plot image
		ax.imshow(images[i].reshape(img_shape),cmap='binary')

		#show true and predicted classes
		if cls_pred is None:
			xlabel = 'True:{0}'.format(cls_true[i])
		else:
			xlabel = 'True:{0},Pred{1}'.format(cls_true,cls_pred)

		ax.set_xlabel(xlabel)

		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images = images,cls_true = cls_true)

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev = 0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05,shape=[length]))

"""This function creates a new convolutional layer in the computational graph for TensorFlow. Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph.

It is assumed that the input is a 4-dim tensor with the following dimensions:

    Image number.
    Y-axis of each image.
    X-axis of each image.
    Channels of each image.
"""

"""
The output is another 4-dim tensor with the following dimensions:

    Image number, same as input.
    Y-axis of each image. If 2x2 pooling is used, then the height and width of the input images is divided by 2.
    X-axis of each image. Ditto.
    Channels produced by the convolutional filters.
"""
def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
	#shape pf filter weights for the convolution
	shape = [filter_size1,filter_size1,num_input_channels,num_filters]
	weights = new_weights(shape=shape)
	biases = new_biases(length=num_filters)

	layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
	layer += biases
	if use_pooling:
		#This is a 2X2 max-pooling ,ie:we select ehe largest value in a 2X2 window
		layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],padding='SAME')
	layer = tf.nn.relu(layer)
	return layer,weights

#Helper function to create a new fully-connected layer
def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):
	weights = new_weights(shape=[num_inputs,num_outputs])

	biases = new_biases(length=num_outputs)
	layer = tf.matmul(input,weights)+biases

	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
"""The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images, img_height, img_width, num_channels].
Note that img_height == img_width == img_size and num_images can be inferred automatically by using -1 for the size of the first dimension.
"""
x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true = tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')

y_true_cls = tf.argmax(y_true,dimension=1)

#Concolutional layer 1
layer_conv1,weights_conv1 = new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size = filter_size1,num_filters=num_filters,use_pooling = True)
layer_conv2,weights_conv2 = new_conv_layer(input=layer_conv1,num_input_channels=num_filters,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)

#Flatten Layer
layer_flat,num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_relu=False)

#prediction class
y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred,dimension=1)

#lets optimize the cost function
#cross entropy optimization
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#optimization Measures
correct_prediction = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#tensorflow run session
session = tf.Session()

session.run(tf.global_variables_initializer())

#Helper function to perform optimization iterations
train_batch_size = 64

total_iterations = 0

def optimize(num_iter):
	global total_iterations
	start_time = time.time()
	for i in range(total_iterations,total_iterations+num_iter):
		x_batch,y_true_batch = data.train.next_batch(train_batch_size)
		feed_dict_train = {x:x_batch,y_true:y_true_batch}
		session.run(optimizer,feed_dict = feed_dict_train)
		if i%100 == 0:
			acc = session.run(accuracy,feed_dict=feed_dict_train)
			msg = "Optimization Iteration:{0:>6},Training Accuracy:{1:>6.1%}"
			print(msg.format(i+1,acc))

	total_iterations += num_iter
	end_time = time.time()
	time_dif = end_time - start_time

	print("Time usage:"+str(timedelta(seconds=int(round(time_dif)))))

#helper function to plot example errors

def plot_example_errors(cls_pred,correct):
	incorrect = (correct==False)
	images = data.test.images[incorrect]
	cls_pred = cls_pred[incorrect]
	cls_true = data.test.cls[incorrect]
	plot_images(images=images[0:9],cls_true=cls_true[0:9],cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

   # Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):

    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

print_test_accuracy()
optimize(num_iterations=99)
print_test_accuracy(show_example_errors=True)