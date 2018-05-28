# This script performs preliminary filtering of the original Instagram data set
import pandas as pd
import csv

print 'Importing the file'
df = pd.read_csv('geosome-instagram.tsv', sep='\t')

df = df.drop(columns=['hashtags', 'photourl'])
print 'Removed hashtagsand photourl columns'

df = df[df.text != '\N']
print 'Removed posts without captions'

df.index = range(len(df))

print 'Extracting hashtags'
hashtags = []
rows_count = float(len(df))
percentage = 0
print str(percentage) + '%'
for index, row in df.iterrows():
    caption = row['text']
    parts = caption.split('#')
    parts.pop(0)
    hashtags_string = ''
    for part in parts:
        hashtags_string += part.split(' ')[0] + ';'
    hashtags.append(hashtags_string[:-1])
    new_percentage = int(index/rows_count * 100)
    if new_percentage != percentage:
        percentage = new_percentage
        print str(percentage) + '%'

df['hashtags'] = pd.Series(hashtags, index=df.index)
print 'Added a hashtags column'

print 'Saving to a file'
df.to_csv('instagram.tsv', sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)

