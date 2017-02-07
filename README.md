# Using convolutional neural networks to rate photo quality

## The data

This project came about because of idle speculation while out hiking: I take loads of photos, and it's often more time consuming to sort through them all at home than it is to take them in the first place. Maybe machine learning can help with this? Given the right data, can I make a model that will tell me which of my photos are better than the others?

I realized that I had come across some data that might work for this: on the British climbing website http://www.ukclimbing.com/ , users upload their climbing and mountaineering photos to share them with the climbing community. Other users can rate the photos from 1\* to 5\* and comment on them.

(upload screenshots of UKC photo page and ratings bit)

At present there are over 165,400 photos shared on UKClimbing.com, and the vast majority of them can be voted on by users. The photos are predominantly landscapes and action shots of climbers, similar to the subjects I mostly take photos of myself. I wrote code to scrape the website and download the small thumbnail images (150x105 pixels), to keep file sizes manageable. I also collected the average vote for each photo (UKC records this as an integer from 1-5, with 5 being the best), the number of votes received, and the date the photo was uploaded.

Number of photos
Number of votes
Vote distribution
Vote increase over time

## The experimental setup

## The neural network architecture

## The results

This code takes thumbnail images scraped from the UK Climbing website (http://www.ukclimbing.com/photos/) which have been rated for photo quality by site users. These photos are used as training data to train neural networks to recognise good and bad images, using the Tensorflow package for Python.
