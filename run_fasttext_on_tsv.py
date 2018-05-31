from nltk.tokenize.punkt import PunktSentenceTokenizer
from urllib.parse import urlparse
import argparse
import fastText
import emoji
import numpy as np
import pandas as pd
import re


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
    """Tokenizes sentences using NLTK's Punkt tokenizer.

    Args:
        caption: A string containing UTF-8 encoded text.

    Returns:
        A list of tokens (sentences).
    """

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

        # Make predictions
        predictions = model.predict(sentences)

        # Get predicted languages and their probabilities
        languages = [p[0].replace('__label__', '') for p in predictions[0]]
        probabilities = [p[0] for p in predictions[1]]

        # Combine languages and predictions
        combined = dict(zip(languages, probabilities))

        # Remove low probabilities
        combined = {k: v for k, v in combined.items() if v > 0.7}

        # If predictions exist, return the output
        if len(combined) > 0:

            # Return set of languages
            return '+'.join(set(combined.keys()))

        else:
            pass

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define the path to input file
ap.add_argument("-i", "--input", required=True,
                help="Path to the input file.")

# Parse arguments
args = vars(ap.parse_args())

# Load language identification model
model = fastText.load_model('models/lid.176.bin')

# Load Instagram data
input_df = pd.read_csv(args['input'], sep='\t', encoding='utf-8')

# Remove the rows that contain only \N
input_df = input_df[input_df.text != r'\N']

# Get predictions for language identification
input_df['langid'] = input_df['text'].apply(lambda x: detect_lang(x))

# Save DataFrame to disk
# input_df.to_pickle(args['output'])

print(input_df['langid'].value_counts()[:20])

