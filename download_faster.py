# this script downloads many URLs using multiple threads
# this is the script for downloading the images
# downloaded and adapted from https://code.google.com/p/workerpool/wiki/MassDownloader

# end 2016: 287231
# end 2015:
# end 2014: 250624

from urllib2 import urlopen
import urllib2
import workerpool
import time
from urllib import urlretrieve
from PIL import Image
import re
import csv
import threading


class DownloadJob(workerpool.Job):
    "Job for downloading a given photo id."
    def __init__(self, photo_id, writer, lock):
        self.photo_id = photo_id
        self.writer = writer
        self.lock = lock

    def run(self):
        result = get_all_data_for_one_photo(self.photo_id)
        if result:
            self.lock.acquire()
            self.writer.writerow(result)
            self.lock.release()


def get_all_data_for_one_photo(photo_id):

    # for one photo id get thumbnail, rating, number of votes
    base_url = 'http://www.ukclimbing.com/images/dbpage.html?id=' + str(photo_id)

    # open url
    try:
        html = urlopen(base_url).read()
    except urllib2.HTTPError:
        return

    category = html.find('/photos/?category')
    votes = html.find('ratingCount')

    # only download photos with a category and that have received votes

    if category > 0 and votes > 0:
        cat_start = html.find('=', category)
        cat_end = html.find('">', category)
        cat_result = int(html[cat_start+1:cat_end])

        # 1 = trad climbing
        # 3 = landscape
        # 14 = winter walking

        # get the star rating (this is always a single digit)
        rating = html.find('ratingValue')
        rating_result = int(html[rating+22])

        # get the number of votes
        votes_start = html.find('>', votes)
        votes_end = html.find('</b>', votes)
        num_votes_result = int(html[votes_start+1:votes_end])

        # get original image size
        if re.search('http://cdn.ukc2.com/i/'+str(photo_id)+'.jpg(\'|\") width', html):
            size_find = re.search('http://cdn.ukc2.com/i/'+str(photo_id)+'.jpg(\'|\") width', html)
            size = size_find.start(0)
            width = html.find('width', size)
            height = html.find('height', size)
            height_end = html.find('alt', height)
            orig_width_result = int(html[width+7:height-2])
            orig_height_result = int(html[height+8:height_end-2])
        else:
            orig_width_result = None
            orig_height_result = None

        # download the thumbnail file
        thumb_url = 'http://cdn.ukc2.com/t/'+str(photo_id)+'.jpg'
        file_name = str(photo_id) + '.jpg'
        urlretrieve(thumb_url, file_name)

        try:
            # get the image dimensions
            im = Image.open(file_name)
            thumb_width_result = im.size[0]
            thumb_height_result = im.size[1]

            return photo_id, cat_result, rating_result, num_votes_result, orig_width_result, orig_height_result, \
                thumb_width_result, thumb_height_result
        except:
            return


def make_pool_and_write_csv(start_id, end_id, output_file):

    # initialize a pool, 10 threads in this case
    pool = workerpool.WorkerPool(size=10)

    # make a lock
    lock = threading.Lock()

    # open csv file, write header row
    with open(output_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['Photo ID', 'Category', 'Rating', 'Number of votes', 'Original width', 'Original height',
                         'Thumbnail width', 'Thumbnail height'])

        # loop through urls and create a job to download each URL
        for id in range(start_id, end_id+1):
            job = DownloadJob(id, writer, lock)
            pool.put(job)

        # send shutdown jobs to all threads, and wait until all the jobs have been completed
        pool.shutdown()
        pool.wait()


start = time.time()
#make_pool_and_write_csv(0, 269787)
make_pool_and_write_csv(269787, 287231, 'results_2.csv')

end = time.time()
print end-start


