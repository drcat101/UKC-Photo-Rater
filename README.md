# Using convolutional neural networks to rate photo quality

This repository contains code to explore whether neural networks can rate photos by their aesthetic quality. Code is provided to scrape a dataset of photos labelled by their quality, and apply a range of neural network architectures in Tensorflow to predict the rating labels directly from the photos.

## The data

The British rock climbing website [UKClimbing](http://www.ukclimbing.com/) contains an ideal dataset for exploring whether neural networks can rate photos according to their aesthetic quality. Users upload their climbing and mountaineering photos to share them with the climbing community, and other users can rate the photos from 1\* to 5\* and comment on them.

At present there are over 165,400 photos shared on UKClimbing.com, and the vast majority of them can be voted on by users. The photos are predominantly landscapes and action shots of climbers. The code in download_faster.py scrapes the website and downloads the small thumbnail images (150x105 pixels). It also collects the average vote for each photo (UKC records this as an integer from 1-5, with 5 being the best), the number of votes received, and the date the photo was uploaded.

Up to the end of 2016, 135,264 thumbnail photos were downloaded. Photos were only downloaded if they had received at least one vote, and the total number of recorded votes for this set of photos is 1,143,878 - over 8 votes per photo, on average.

The machine learning system is set up as a binary classification problem, with class 1 as photos rated 5\* (the best) and class 0 as photos rated 1\* and 2\*. Voting is much more likely to be consistent in these classes - people are more likely to agree on very good vs very bad photos.


## The neural network architecture

- based on https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
- first tried a simple logistic regression (tf_binary.py)
- then a neural network with a single hidden layer (tf_hidden.py)
- then a convolutional neural network on photos in black and white
- then a convnet in color



