# merging code from here:
# https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
# with code from here: https://github.com/jasonbaldridge/try-tf/blob/master/hidden.py
# converts photos to black and white to reduce file sizes

import tensorflow as tf
import time
import os
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import sys


# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.

tf.app.flags.DEFINE_integer('number_epochs', 1,
                            'Number of passes over the training data.')
tf.app.flags.DEFINE_integer('num_hidden', 1,
                            'Number of nodes in the hidden layer.')
tf.app.flags.DEFINE_boolean('verb', False, 'Produce verbose output.')
FLAGS = tf.app.flags.FLAGS


# adapt vectorize_image function so that it takes an average of all 3 channels

def vectorize_image_bw(photo_id):
    # this turns each image into a 1x47250 vector
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = 'downloaded_thumbnails/' + str(photo_id) + '.jpg'
    abs_file_path = os.path.join(script_dir, rel_path)
    try:
        im = Image.open(abs_file_path)
        im_list = list(im.getdata())
        # problem if the image is in b&W - list of ints instead of tuples
        # tf can't use vectors of different sizes - so skip it
        if type(im_list[0]) == int:
            return
        else:
            im_vector = []
            for tup in im_list:
                pix = (float(tup[0]) + float(tup[1]) + float(tup[2]))/3
                im_vector.append(pix/255.0)
            # need to have all vectors same length
            # some photos are different sizes
            if len(im_vector) == 15750:
                return im_vector
            else:
                return
    except IOError:
        return


def extract_data(start_id, end_id, num_images):
    # this selects a specified number of images rated 0, 1 or 4
    # open CSV file with labels
    results_df = pd.read_csv('results.csv')
    results_df.columns = [c.replace(' ', '_') for c in results_df.columns]

    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # keep a record of which photos are chosen
    id_list = []

    # Get features and vectors as lists
    count = 0
    count_good = 0
    count_bad = 0


    for i in range(start_id, end_id +1):
        if count < num_images:
            try:
                y_row = results_df[results_df.Photo_ID == i]

                if int(y_row.iloc[0, 1]) != 4:  # take out indoor climbing photos
                    # get the photo rating
                    # use rating from 0 to 4 instead of 1 to 5
                    y_i = int(y_row.iloc[0, 2])-1
                    if y_i == 0 or y_i == 1:
                      #  print y_i
                        if count_bad < num_images/2:
                            X_i = vectorize_image_bw(i)
                            if X_i:
                                fvecs.append(X_i)
                                labels.append(0)
                                id_list.append(i)
                                count += 1
                                count_bad += 1
                    elif y_i == 4:
                       # print y_i
                        if count_good < num_images/2:
                            X_i = vectorize_image_bw(i)
                            if X_i:
                                labels.append(1)
                                fvecs.append(X_i)
                                id_list.append(i)
                                count += 1
                                count_good += 1
            except IndexError:
                continue


    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)
#    print labels

    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)
#    return labels_onehot
#    print id_list
    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np, labels_onehot


def make_and_pickle_vectorized_photos(start_num, end_num, num_photos):
    features, values = extract_data(start_num, end_num, num_photos)
    with open('objs_bw.pickle', 'w') as f:
        pickle.dump([features, values], f)

#make_and_pickle_vectorized_photos(100000, 250000, 10000)

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

    # Load extracted data in numpty arrays from pickle
    with open('objs_bw.pickle') as f:
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
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to
    # image width and height, and the final dimension corresponding to the number of color channels.
    x_image = tf.reshape(x, [-1, 150, 105, 1])

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

    # Finally, we add a softmax layer, just like for the one layer softmax regression.

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
            sys.stdout.write('.')
            sys.stdout.flush()
            if step % 10 == 0:
                print("test accuracy %g"%accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

        print("test accuracy %g"%accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

if __name__ == '__main__':
    tf.app.run()





