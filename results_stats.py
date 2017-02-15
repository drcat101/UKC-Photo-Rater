import csv
import seaborn as sns


def count_votes_in_csv(csvfile):
    count = 0
    votes = 0
    max_votes = 0

    with open(csvfile, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            count += 1
            votes += float(row[3])
            if float(row[3]) > max_votes:
                max_votes = float(row[3])

    return int(votes), votes/count, int(max_votes)


# print count_rows_in_csv('results.csv')
# 128773 photos, 1083333 votes, average 8.41 votes per photo
# largest number of votes for one photo is 720
# photos only downloaded with at least 1 vote

#print count_votes_in_csv('results_all.csv')
# results_all is up to the end of 2016

def count_ratings_and_categories(csvfile):

    rating_dict = dict([('1', 0), ('2',0), ('3', 0), ('4', 0), ('5', 0), ('No rating', 0)])
    category_dict = dict([('1', 0), ('2',0), ('3', 0), ('4', 0), ('11', 0), ('12', 0), ('13', 0), ('14', 0), ('15', 0), ('16', 0),
                          ('17', 0), ('18', 0), ('No category', 0)])

    with open(csvfile, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the header row

        for row in reader:
            rating = str(row[2])
            category = str(row[1])

            try:
                rating_dict[rating] += 1
            except KeyError:
                rating_dict['No rating'] += 1
            try:
                category_dict[category] += 1
            except KeyError:
                category_dict['No category'] += 1

    return rating_dict, category_dict

#print count_ratings_and_categories('results.csv')
# {'1': 2454, 'No rating': 1, '3': 39874, '2': 11627, '5': 22047, '4': 52770}
# {'11': 14015, '13': 11437, '12': 14515, '15': 93, '14': 2606, '17': 1147, '16': 244, '18': 990, '1': 37880,
# '3': 32804, '2': 10982, '4': 2017, 'No category': 43})


print count_ratings_and_categories('results_all.csv')
# {'1': 2479, 'No rating': 1, '3': 42146, '2': 12017, '5': 23188, '4': 55433}
