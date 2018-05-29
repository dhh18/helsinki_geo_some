# -*- coding: latin-1 -*-

# This script does the following
# 1. drops rows where 'text' is empty
# 2. drops the 'photourl' column
# 3. extracts hashtags from 'text' column to a seperate  column named 'ext_hashtags'
# 4. runs language detection on 'text' column and adds a new column called 'langid'
# 5. drops rows where 'langid' is empty
# 6. saves it into lang_instagram.tsv

from nltk.tokenize.punkt import PunktSentenceTokenizer
try:
    from urllib.parse import urlparse
except ImportError:
     from urlparse import urlparse
import argparse
import fastText
import emoji
import numpy as np
import pandas as pd
import re
import csv
import geopandas as gpd
from shapely.geometry import *
import geojson
import io
 
 #### Tuomo's code  from https://www.dropbox.com/s/3982fdy56zn7ggw/run_fasttext.py starts

# Define the preprocessing function
def preprocess_caption(row, mode):
    """Applies the selected preprocessing mode to a row of text.
 
     Args:
         row: A UTF-8 string.
         mode: A string indicating the selected preprocessing strategy.
 
     Returns:
         A string containing the preprocessed row.
    """
    # Check if preprocessing has been requested.
    if mode != 'no_preprocessing':
 
        # Convert unicode emoji to shortcode emoji
        row = emoji.demojize(row)
        # Remove single emojis and their groups
        row = re.sub(r':(?<=:)([a-zA-Z0-9_\-&\'’]*)(?=:):', '', row)
 
        # Apply the selected preprocessing strategy defined in the variable
        # 'mode'. This defines how each row is processed. The selected mode
        # defines the preprocessing steps applied to the data below by
        # introducing different conditions.
 
        # Remove all mentions (@) in the caption
        row = re.sub(r'@\S+ *', '', row)
        # If mode is 'rm_all', remove all hashtags (#) in the caption
        if mode == 'rm_all':
            row = re.sub(r'#\S+ *', '', row)
 
        # Split the string into a list
        row = row.split()
        # Remove all non-words such as smileys etc. :-)
        row = [word for word in row if re.sub('\W', '', word)]
        # Check the list of items for URLs and remove them
        row = [word for word in row if not urlparse(word).scheme]
        # Attempt to strip extra linebreaks following any list item
        row = [word.rstrip() for word in row]
 
        # If mode is 'rm_trail', remove hashtags trailing the text, e.g.
        # "This is the caption and here are #my #hashtags"
        if mode == 'rm_trail':
            while len(row) != 0 and row[-1].startswith('#'):
                row.pop()
 
        # Reconstruct the row
        row = ' '.join(row)
 
        # If mode is 'rm_trail', drop hashes from any remaining hashtags
        if mode == 'rm_trail':
            row = re.sub(r'g*#', '', row)
 
    # Simplify punctuation, removing sequences of exclamation and question
    # marks, commas and full stops, saving only the final character
    row = re.sub(r'[?.!,_]+(?=[?.!,_])', '', row)
 
    # Return the preprocessed row
    return row
 
 
def split_sentence(caption):
   
    # Initialize the sentence tokenizer
    tokenizer = PunktSentenceTokenizer()
    # Tokenize the caption
    caption_tokens = tokenizer.tokenize(caption)
    # Return a list of tokens (sentences)
    return caption_tokens
 
 
def detect_lang(caption):
    # If the caption is None, return None
    if caption == 'None' or caption is None:
        return
 
    # Preprocess the caption
    caption = preprocess_caption(caption, 'rm_all')
 
    # Perform sentence splitting for any remaining text
    if len(caption) == 0:
        return None
 
    else:
        # Get sentences
        sentences = split_sentence(caption)
 
        # Calculate the character length of each sentence
        char_len = [len(s) for s in sentences]
        # Make predictions
        predictions = model.predict(sentences)
 
        # Get predicted languages and their probabilities
        languages_unicoded = [p[0].replace('__label__', '') for p in predictions[0]]
        languages = map(str, languages_unicoded)
        probabilities = [p[0] for p in predictions[1]]

        #Return the language only if there is no ambiguity (meaning same caption having detected as multiple langues), probability is more than 42% and character length is at least 10

        lang = None
        if len(languages) == len(set(languages)):
        	if len(set(languages)) == 1:
        		if char_len >= 10:
        			if probabilities > 0.42 :
        				lang = languages[0]

        else:
        	pass
        

        return lang


#### Tuomo's code  from https://www.dropbox.com/s/3982fdy56zn7ggw/run_fasttext.py ends


def extract_hashtags(s):
	return set(part[1:] for part in s.split() if part.startswith('#'))

df = pd.read_csv("geosome_instagram.tsv", na_values=['', ' ', '\N'], encoding='latin-1', sep="\t")
initial_rows = len(df)

# Find how many blank cells in each columns
null_columns=df.columns[df.isnull().any()]
print df[null_columns].isnull().sum()

rows_with_text_empty = df['text'].isnull().sum()


# Replace empty spaces with NaN so that we can drop them later with dropna()
df = df.applymap(lambda x: np.nan if isinstance(x, basestring) and x.isspace() else x)

df.dropna(subset=['text'], inplace=True)


if initial_rows - rows_with_text_empty == len(df):
	print "\n Out of {} rows, {} rows were found with empty text field. They were dropped, and the number of rows now is {}".format(initial_rows, rows_with_text_empty, len(df))
else:
	print "Check again!"

df.drop('photourl', axis=1, inplace=True)


# Need to check the hashtag extraction function. Sid is not sure about the accuracy!
df['ext_hashtags']= df.apply(lambda row: extract_hashtags(row['text']), axis=1)
#print list(df.columns.values)

print("Importing Helsinki polygon data")
json_file = io.open('helsinki.json', 'r', encoding='utf-8')  # Reading the GeoJSON file
gson = geojson.loads(json_file.read())
gdf = gpd.GeoDataFrame.from_features(gson)  # Creating a data frame from the GeoJSON data
gdf.crs = {'init': 'epsg:3879'}  # Setting the source coordinate system
gdf.geometry = gdf.geometry.to_crs(epsg=4326)  # Converting to the traditional coordinate system
helsinki_poly = gdf.loc[1, 'geometry'] # Extracting the Helsinki region polygon

print("Performing point-in-polygon matching")
withinHelsinki = []
rows_count = float(len(df))
for index, row in df.iterrows():
    lat = float(row['lat'])
    lon = float(row['lon'])
    point = Point(lon, lat)
    withinHelsinki.append(helsinki_poly.intersects(point))  # Checking whether a point lies within the Helsinki polygon
    
    
df['within_Helsinki'] = pd.Series(withinHelsinki, index=df.index) # Adding the within_Helsinki column to the data frame

# Load language identification model
model = fastText.load_model('fastText_models/lid.176.bin')
 
 
# Get predictions for language identification
print "Performing language detection"
df['langid'] = df['text'].apply(lambda x: detect_lang(x))

print "Successfully managed to detect languages for {} rows".format(len(df)-df['langid'].isnull().sum())

#drop rows if langid was not successfull
df.dropna(subset=['langid'], inplace=True)


language_diversity_df = df['langid'].value_counts()
language_diversity_df.to_csv('language_diversity.csv', sep='\t')
print "Count of successfully detected languages are saved in language_diversity.csv"
 
# Save DataFrame to disk
print 'Saving processed dataframe to instagram_processed.pkl as well as to a .tsv file named instagram_processed.tsv'
df.to_csv('instagram_processed.tsv', sep='\t', quoting=csv.QUOTE_NONNUMERIC, encoding='latin-1', index=False)
df.to_pickle('instagram_processed.pkl')
