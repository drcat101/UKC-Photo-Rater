# merging code from here: https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
# with code from here: https://github.com/jasonbaldridge/try-tf/blob/master/hidden.py
# trying colour photos this time

import tensorflow as tf
import time

import pickle
import sys
from tensorflow_binary import extract_data, vectorize_image


# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.

tf.app.flags.DEFINE_integer('number_epochs', 10,
                            'Number of passes over the training data.')

tf.app.flags.DEFINE_boolean('verb', True, 'Produce verbose output.')
FLAGS = tf.app.flags.FLAGS



def make_and_pickle_vectorized_photos(start_num, end_num, num_photos):
    features, values = extract_data(start_num, end_num, num_photos)
    with open('objs_col_feb_10k.pickle', 'w') as f:
        pickle.dump([features, values], f)

#make_and_pickle_vectorized_photos(269787, 287231, 10000)

from tensorflow_binary import make_training_testing_data


# helper functions to initialize weights and biases
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# helper functions for convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(argv=None):
    start = time.time()

    # Be verbose?
    verbose = FLAGS.verb

    # Load extracted data in numpy arrays from pickle
    with open('objs_col.pickle') as f:
        features, values = pickle.load(f)
    print 'Finished loading data'

    train_data, test_data, train_labels, test_labels = make_training_testing_data(features, values)

    # Get the shape of the training data.
    train_size, num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.number_epochs

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])

    # Define and initialize the network.

    # Initialize the weights and biases.
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    # To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to
    # image width and height, and the final dimension corresponding to the number of color channels.
    x_image = tf.reshape(x, [-1, 150, 105, 3])

    # We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # In order to build a deep network, we stack several layers of this type.
    # The second layer will have 64 features for each 5x5 patch.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer
    W_fc1 = weight_variable([38 * 27 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 38 * 27 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # To reduce overfitting, we will apply dropout before the readout layer.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Finally, we add a softmax layer

    W_fc2 = weight_variable([1024, NUM_LABELS])
    b_fc2 = bias_variable([NUM_LABELS])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.Session() as s:
        print 'Starting training'
        cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        s.run(tf.initialize_all_variables())
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})
            sys.stdout.write(str(step)+', ')
            sys.stdout.flush()
            if step % 20 == 0:
                print("Current test accuracy: %g"%accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

        print("Final test accuracy: %g"%accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

if __name__ == '__main__':
    tf.app.run()





