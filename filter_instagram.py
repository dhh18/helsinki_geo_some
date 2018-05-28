# This script performs preliminary filtering of the original Instagram data set
import pandas as pd
import csv
from string import punctuation

def extract_hashtag(caption_part, delimiters):
    part = caption_part
    for delimiter in delimiters:
        part = part.split(delimiter)[0]
    return part

print 'Importing the file'
df = pd.read_csv('geosome-instagram.tsv', sep='\t')

df = df.drop(columns=['photourl'])
print 'Removed the photourl column'

df = df[df.text != '\N']
print 'Removed posts without captions'

df.index = range(len(df))

print 'Extracting hashtags'
hashtags = []
rows_count = float(len(df))
percentage = 0
delimiters = ' ' + punctuation.replace('#', '')
print str(percentage) + '%'
for index, row in df.iterrows():
    caption = row['text'].replace('@', '')
    parts = caption.split('#')
    local_hashtags = []
    for part in parts[1:]:
        local_hashtags.append(extract_hashtag(part, delimiters))
    caption = ''.join(parts)
    for hashtag in local_hashtags:
        caption = caption.replace(hashtag, '')
    df.at[index, 'text'] = caption
    df.at[index, 'hashtags'] = ';'.join(local_hashtags)
    new_percentage = int(index/rows_count * 100)
    if new_percentage != percentage:
        percentage = new_percentage
        print str(percentage) + '%'

print 'Saving to a file'
df.to_csv('instagram.tsv', sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)

