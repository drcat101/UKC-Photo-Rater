# this is the script for training the model in Tensorflow, saving the model, running a saved model on new data,
# binary classifier, no hidden layers (i.e. logistic regressor)

import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import os
from sklearn.cross_validation import train_test_split
import pickle

# Global variables.
NUM_LABELS = 2   # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.
PICKLE_FILE = 'objs_col.pickle'
START_NUM = 240000
END_NUM = 265000
NUM_PHOTOS = 5000



# Define the flags usable from the command line.
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of passes over the training data')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
tf.app.flags.DEFINE_string('model_file', None, 'Filename for the saved model')
FLAGS = tf.app.flags.FLAGS


def vectorize_image(photo_id):
    # this turns each image into a 1x47250 vector
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = 'downloaded_thumbnails/' + str(photo_id) + '.jpg'
    abs_file_path = os.path.join(script_dir, rel_path)
    try:
        im = Image.open(abs_file_path)
        im_list = list(im.getdata())
        im_vector = []

        # problem if the image is in b&W - list of ints instead of tuples
        # need all vectors to be the same size - so make tuples that are the int repeated 3 times
        if type(im_list[0]) == int:
            for i in im_list:
                i = float(i)
                threei = [i]*3
                im_vector.extend(threei)
            if len(im_vector) == 47250:
                return im_vector
            else:
                return
        else:
            im_vector = []
            for tup in im_list:
                for i in tup:
                    i = float(i)
                    im_vector.append(i/255.0)
            # need to have all vectors same length
            # some photos are different sizes
            if len(im_vector) == 47250:
                return im_vector
            else:
                return
    # if image doesn't exist
    except IOError:
        return


# only considering bad images (rating 0 or 1) versus good images (rated 4)

def extract_data(start_id, end_id, num_images):
    # this selects a specified number of images rated 0, 1 or 4
    # open CSV file with labels
    results_df = pd.read_csv('results_all.csv')
    results_df.columns = [c.replace(' ', '_') for c in results_df.columns]

    # arrays to hold the labels and feature vectors.
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
                    # so bad = 0 or 1 and good = 4
                    y_i = int(y_row.iloc[0, 2])-1
                    if y_i == 0 or y_i == 1:
                        if count_bad < num_images/2:
                            X_i = vectorize_image(i)
                            if X_i:
                                fvecs.append(X_i)
                                labels.append(0)
                                id_list.append(i)
                                count += 1
                                count_bad += 1
                    elif y_i == 4:
                        if count_good < num_images/2:
                            X_i = vectorize_image(i)
                            if X_i:
                                labels.append(1)
                                fvecs.append(X_i)
                                id_list.append(i)
                                count += 1
                                count_good += 1
            except IndexError:
                continue

    # convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    # convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    # return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np, labels_onehot


#print extract_data(244000, 245000, 10)

def make_and_pickle_vectorized_photos(start_num, end_num, num_photos):
    features, values = extract_data(start_num, end_num, num_photos)
    with open(PICKLE_FILE, 'w') as f:
        pickle.dump([features, values], f)

#make_and_pickle_vectorized_photos(START_NUM, END_NUM, NUM_PHOTOS)


def make_training_testing_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def main(argv=None):
    # adapted from blog post here: https://bcomposes.wordpress.com/2015/11/26/simple-end-to-end-tensorflow-examples/

    # Be verbose?
    verbose = True

    # Extract it into numpy matrices.
    # Put start photo ID, end photo ID and number of images here.

    with open(PICKLE_FILE) as f:
        features, values = pickle.load(f)
    train_data, test_data, train_labels, test_labels = make_training_testing_data(features, values)

    # get the number of images in each category
    test_labels_dict = {'0': 0, '1': 0}
    test_labels_length, _ = np.shape(test_labels)
    for i in test_labels:
        itemindex = np.where(i == 1)
        dict_index = str(itemindex[0][0])
        test_labels_dict[dict_index] += 1
    for key in test_labels_dict:
        test_labels_dict[key] = test_labels_dict[key]/float(test_labels_length)
    print test_labels_dict
    print test_labels_length


    # Get the shape of the training data.
    train_size, num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])

    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # Define and initialize the network.

    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([num_features, NUM_LABELS]))
    b = tf.Variable(tf.zeros([NUM_LABELS]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()

        # make thing to save it
        saver = tf.train.Saver()

        if verbose:
            print 'Initialized!'
            print
            print 'Training.'

        # Iterate and train.
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print step,
                print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})

            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})

            if verbose and offset >= train_size-BATCH_SIZE:
                print

        # Give very detailed output.
        if verbose:
            print
            print 'Weight matrix.'
            print s.run(W)
            print
            print 'Bias vector.'
            print s.run(b)
            print
            print "Applying model to first test instance."
            first = test_data[:1]
            print "Point =", first
            print "Wx+b = ", s.run(tf.matmul(first,W)+b)
            print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first,W)+b))
            print
            print 'Argmax'
            p = tf.argmax(y, 1)
            print(s.run(p, feed_dict={x: test_data, y_: test_labels}))

        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})

        #save_filename = FLAGS.model_file + '.ckpt'
        #save_path = saver.save(s, save_filename)
        #print "Model saved in file: ", save_path




if __name__ == '__main__':
    tf.app.run()

# python tensorflow_binary.py

# 10,000 photos, 50:50 split 1&2 vs 5, 100 epochs, learning rate 0.0001, AdamOptimizer: accuracy 0.621


def from_saved_model(model_path):
    test_data, test_labels = extract_data(100000, 250000, 10000)

    # get the shape of the training data.
    _, num_features = test_data.shape

    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])

    W = tf.Variable(tf.zeros([num_features, NUM_LABELS]))
    b = tf.Variable(tf.zeros([NUM_LABELS]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # add options to save and restore all the variables.
    saver = tf.train.Saver()

    rating_list = []

    # later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, model_path)
        print "Model restored."

        first = test_data[:1]
        print "Point =", first
        print "Wx+b = ", sess.run(tf.matmul(first, W)+b)
        print "softmax(Wx+b) = ", sess.run(tf.nn.softmax(tf.matmul(first,W)+b))

        for i in test_data:
            rating = sess.run(tf.nn.softmax(tf.matmul(i, W)+b))
            rating_list.append(rating[0][1])
        return rating_list

#rating_list = from_saved_model('trained_model_20160131.ckpt')
#id_list = [100004, 100010, 100012, 100051, 100064, 100065, 100076, 100079, 100080, 100094, 100099, 100116, 100122, 100142, 100158, 100167, 100168, 100171, 100184, 100186, 100193, 100212, 100214, 100215, 100217, 100218, 100219, 100220, 100224, 100229, 100241, 100244, 100249, 100250, 100257, 100263, 100264, 100269, 100275, 100276, 100286, 100305, 100308, 100310, 100328, 100349, 100360, 100362, 100381, 100385, 100396, 100403, 100406, 100437, 100447, 100455, 100457, 100469, 100477, 100484, 100494, 100495, 100517, 100638, 100639, 100647, 100650, 100696, 100726, 100760, 100793, 100796, 100803, 100819, 100888, 100917, 100927, 100939, 100959, 100973, 100996, 101171, 101248, 101333, 101334, 101341, 101381, 101383, 101384, 101385, 101386, 101387, 101458, 101532, 101588, 101589, 101619, 101625, 101628, 101637]
#true_ratings = [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#ratings_ids = zip(id_list, rating_list, true_ratings)
#print ratings_ids
#print sorted(ratings_ids, key=lambda x: x[1])
#sorted_ratings_ids = sorted(ratings_ids, key=lambda x: x[1])
#print sorted_ratings_ids[:10]
#print sorted_ratings_ids[-5:]
#print sorted_ratings_ids[44:54]
# strong true positives: 100305, 100224, 100257
# strong true negatives: 101458, 100639, 100939, 100917, 100264, 100696, 101385, 100927
# middle: incorrectly rated 100517 (false positive), 100888 (false positive), correctly rated 100064(weak positive)
# [(100888, 0.57268542, 0), (100064, 0.57562107, 1), (101589, 0.58495629, 0), (100647, 0.58832908, 0),
# (100973, 0.59642172, 0), (101532, 0.60017419, 0), (100517, 0.60316259, 0), (100229, 0.61357808, 1),
# (100010, 0.61579573, 0), (100360, 0.64027715, 1)]

# accuracy with objs_col: 0.631 (2000 photos)