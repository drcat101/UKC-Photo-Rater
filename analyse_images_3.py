# for each photo rated 1
# split into rgb channels
# get histogram 256 bins (numpy)
# add the next one
# do for 100 images
# do for images rated 5
# also try the colorfulness measure: https://gist.github.com/zabela/8539116
# or colorfulness in this: http://infolab.stanford.edu/~wangz/project/imsearch/Aesthetics/ECCV06/datta.pdf



# this script makes visualization 1

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import csv
import numpy as np
import random

# for each photo rated 1
def get_best_and_worst_images(csvfile):
    rated_1_or_2 = []
    rated_5 = []
    with open(csvfile, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if row[1] != '4':  # exclude indoor climbing category
                if row[2] == '1' or row[2] == '2':
                    rated_1_or_2.append(int(row[0]))
                elif row[2] == '5':
                    rated_5.append(int(row[0]))
    print len(rated_1_or_2)
    print len(rated_5)
    return rated_1_or_2, rated_5


def get_rgb_channels(id_list, num_images):
    red, green, blue = [], [], []
    random.seed(6)
    selection = random.sample(id_list, num_images)
    count = 0
    for i in selection:
        while count < num_images:
            filename = 'downloaded_thumbnails/' + str(i) + '.jpg'
            im = Image.open(filename)
            im_list = list(im.getdata())
            if type(im_list[0]) == int:
                continue
            for tup in im_list:
                red.append(tup[0])
                green.append(tup[1])
                blue.append(tup[2])
                count += 1
    return red, green, blue


bad_list, good_list = get_best_and_worst_images('results.csv')


def plot_hists(bad_list, good_list):
    sns.set(style="white", palette="muted", color_codes=True)
    fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True)
    ax0, ax1, ax2 = axes.flat
    axes_list = [ax0, ax1, ax2]
    bad_channels = get_rgb_channels(bad_list, 1000)
    good_channels = get_rgb_channels(good_list, 1000)
    color = ['darkred', 'green', 'steelblue']
    for i in range(3):
        values, bins = np.histogram(bad_channels[i], bins=np.arange(256))
        values_g, bins_g = np.histogram(good_channels[i], bins=np.arange(256))
        line1, = axes_list[i].plot(bins[:-1], values, color='black')
        line1.set_label('Low rated photos')
        axes_list[i].legend()
        axes_list[i].fill_between(bins[:-1], 0, values, alpha=0.7, facecolor=color[i])
        line2, = axes_list[i].plot(bins_g[:-1], values_g, color=color[i])
        line2.set_label('High rated photos')
        axes_list[i].legend()
        axes_list[i].fill_between(bins_g[:-1], 0, values_g, alpha=0.3, facecolor=color[i])
        axes_list[i].set_xlim(xmax=256)
        axes_list[i].set_xlabel('Pixel intensity (0-255)')

    ax0.set_title('Red', fontsize=14, fontweight='bold')
    ax1.set_title('Green', fontsize=14, fontweight='bold')
    ax2.set_title('Blue', fontsize=14, fontweight='bold')
    ax0.set_ylabel('Frequency (1,000s)')


    plt.tight_layout()

    plt.show()

plot_hists(bad_list, good_list)

# this makes a plot of 9 images, used for Data Incubator application
# strong true positives: 100305, 100224, 100257
# strong true negatives: 101458, 100639, 100939, 100917, 100264, 100696, 101385, 100927
# middle: incorrectly rated 100517 (false positive), 100888 (false positive), correctly rated 100064(weak positive)
# [(100888, 0.57268542, 0), (100064, 0.57562107, 1), (101589, 0.58495629, 0), (100647, 0.58832908, 0),
# (100973, 0.59642172, 0), (101532, 0.60017419, 0), (100517, 0.60316259, 0), (100229, 0.61357808, 1),
# (100010, 0.61579573, 0), (100360, 0.64027715, 1)]

def plot_many_images():
    # make 3x3 grid of images
    sns.set(style="white", palette="muted", color_codes=True)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    img1 = Image.open('downloaded_thumbnails/100305.jpg')
    ax1.imshow(img1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title('Strong positive ratings', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Correctly classified as a good image', fontsize=8)

    img2 = Image.open('downloaded_thumbnails/100224.jpg')
    ax4.imshow(img2)
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.set_xlabel('Correctly classified as a good image', fontsize=8)

    img3 = Image.open('downloaded_thumbnails/100257.jpg')
    ax7.imshow(img3)
    ax7.set_yticklabels([])
    ax7.set_xticklabels([])
    ax7.set_xlabel('Correctly classified as a good image', fontsize=8)

    img4 = Image.open('downloaded_thumbnails/100517.jpg')
    ax2.imshow(img4)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title('Weak ratings', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Incorrectly classified as a good image', fontsize=8, color='darkred')

    img5 = Image.open('downloaded_thumbnails/100010.jpg')
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    ax5.imshow(img5)
    ax5.set_xlabel('Incorrectly classified as a good image', fontsize=8, color='darkred')

    img6 = Image.open('downloaded_thumbnails/100229.jpg')
    ax8.set_yticklabels([])
    ax8.set_xticklabels([])
    ax8.imshow(img6)
    ax8.set_xlabel('Correctly classified as a good image', fontsize=8)

    img7 = Image.open('downloaded_thumbnails/101458.jpg')
    ax3.imshow(img7)
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.set_title('Strong negative ratings', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Correctly classified as a bad image', fontsize=8)

    img8 = Image.open('downloaded_thumbnails/100264.jpg')
    ax6.set_yticklabels([])
    ax6.set_xticklabels([])
    ax6.imshow(img8)
    ax6.set_xlabel('Correctly classified as a bad image', fontsize=8)

    img9 = Image.open('downloaded_thumbnails/100939.jpg')
    ax9.set_yticklabels([])
    ax9.set_xticklabels([])
    ax9.imshow(img9)
    ax9.set_xlabel('Correctly classified as a bad image', fontsize=8)

    plt.tight_layout()
    plt.show()

    return

print plot_many_images()



