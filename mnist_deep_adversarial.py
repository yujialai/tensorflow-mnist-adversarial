#import mnist data and tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

##rebuild the softmax regression model
#place holders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

##rebuild the multilayer convolutional network
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#train and evaluate the model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    #restore variaables
    saver.restore(sess, "nets/outputyujinet.ckpt")
    print("---------------------------------------------------------------------")
    print("Model restored.")

    # correct classifying reseult outputed by the trained net
    net_result = y_conv.eval(feed_dict={x:mnist.train.images[10:11], y_:mnist.train.labels[10:11], keep_prob:1.0})
    print "'correct' net's classifying result of x(y_conv) is: ", net_result

    # target is to classify as a 2
    y_target = [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]
    epsilon = .3
    gradient = tf.gradients(cross_entropy, x)
    add = sess.run(epsilon * tf.sign(gradient), feed_dict = {x:mnist.train.images[10:11], y_:y_target, keep_prob:1.0})
    x_ad = mnist.train.images[10:11] - add
    x_ad = x_ad[0]
    print "x_ad generated"
    
    # adversarial image feed into the trained net
    # should give a wrong prediction even though the net is working well
    net_result_ad = y_conv.eval(feed_dict={x:x_ad, y_:y_target, keep_prob:1.0})
    print "fooled net classifying result of x_ad is: ", net_result_ad


label_from_mnist = mnist.train.labels[10:11]
print "the correct classifying result should be: ", label_from_mnist
