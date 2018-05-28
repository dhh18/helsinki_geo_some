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
        languages = [p[0].replace('__label__', '') for p in predictions[0]]
        probabilities = [p[0] for p in predictions[1]]
 
        # Return languages and probabilities
        return list(zip(languages, probabilities, char_len))


#### Tuomo's code  from https://www.dropbox.com/s/3982fdy56zn7ggw/run_fasttext.py ends


def extract_hashtags(s):
	return set(part[1:] for part in s.split() if part.startswith('#'))

df = pd.read_csv("geosome-instagram.tsv", na_values=['', ' ', '\N'], encoding='latin-1', sep="\t")
initial_rows = len(df)

# Find how many blank cells in each columns
null_columns=df.columns[df.isnull().any()]
print df[null_columns].isnull().sum()

rows_with_text_empty = df['text'].isnull().sum()


# Replace empty spaces with NaN so that we can drop them later with dropna()
df = df.applymap(lambda x: np.nan if isinstance(x, basestring) and x.isspace() else x)

df.dropna(subset=['text'], inplace=True)


if initial_rows - rows_with_text_empty == len(df):
	print "Out of {} rows, {} rows were found with empty text field. They were dropped, and the number of rows now is {}".format(initial_rows, rows_with_text_empty, len(df))
else:
	print "Check again!"

df.drop('photourl', axis=1, inplace=True)


print list(df.columns.values)

df['ext_hashtags']= df.apply(lambda row: extract_hashtags(row['text']), axis=1)
print list(df.columns.values)


# Load language identification model
model = fastText.load_model('fastText_models/lid.176.bin')
 
 
# Get predictions for language identification
df['langid'] = df['text'].apply(lambda x: detect_lang(x))

#drop rows if langid was not successfull
df.dropna(subset=['langid'], inplace=True)
 
# Save DataFrame to disk
print 'Saving to a file'
df.to_csv('lang_instagram.tsv', sep='\t', quoting=csv.QUOTE_NONNUMERIC, encoding='latin-1', index=False)
