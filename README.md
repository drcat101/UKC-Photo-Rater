# Using convolutional neural networks to rate photo quality

## The data

This project came about because of idle speculation while out hiking: I take loads of photos, and it's often more time consuming to sort through them all at home than it is to take them in the first place. Maybe machine learning can help with this? Given the right data, can I make a model that will tell me which of my photos are better than the others?

I realized that I had come across some data that might work for this: on the British climbing website http://www.ukclimbing.com/ , users upload their climbing and mountaineering photos to share them with the climbing community. Other users can rate the photos from 1\* to 5\* and comment on them.

(upload screenshots of UKC photo page and ratings bit)

At present there are over 165,400 photos shared on UKClimbing.com, and the vast majority of them can be voted on by users. The photos are predominantly landscapes and action shots of climbers, similar to the subjects I mostly take photos of myself. I wrote code to scrape the website and download the small thumbnail images (150x105 pixels), to keep file sizes manageable. I also collected the average vote for each photo (UKC records this as an integer from 1-5, with 5 being the best), the number of votes received, and the date the photo was uploaded.

The code used to scrape the images is included in this repo as download_faster.py. 135,264 thumbnail photos were downloaded, ranging in upload date from 2000 to the end of 2016. Photos were only downloaded if they had received at least one vote, and the total number of recorded votes for this set of photos is 1,143,878 - over 8 votes per photo, on average.

I set up the machine learning system as a binary classification problem, with class 1 as photos rated 5\* (the best) and class 0 as photos rated 1\* and 2\*. Voting is much more likely to be consistent in these classes - people are more likely to agree on very good vs very bad photos.

- Vote distribution
- Vote increase over time
- 

## The neural network architecture

- based on https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
- first tried a simple logistic regression (tf_binary.py)
- then a neural network with a single hidden layer (tf_hidden.py)
- then a convolutional neural network on photos in black and white
- then a convnet in color

## The results


