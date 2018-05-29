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

df = pd.read_pickle('instagram_processed.pkl')
initial_row_count = len(df)

df_eng = df.loc[df['langid'] == 'en']
rows_with_english = len(df_eng)

df_eng_helsinki = df_eng.loc[df_eng['within_Helsinki'] == True]
rows_with_english_Helsinki  = len(df_eng_helsinki)

print "\nOut of {} rows, {} rows were tagged as English. Then, {} rows corresponds to posts from Helsinki Region. ".format(initial_row_count, rows_with_english, rows_with_english_Helsinki)

print "\nRemoving within_Helsinki column!"
df_eng_helsinki.drop('within_Helsinki', axis=1, inplace=True)
# Save DataFrame to disk
print '\nSaving filtered dataframe to instagram_eng_hel.pkl as well as to a .tsv file named instagram_eng_hel.tsv'
df_eng_helsinki.to_csv('instagram_eng_hel.tsv', sep='\t', quoting=csv.QUOTE_NONNUMERIC, encoding='latin-1', index=False)
df_eng_helsinki.to_pickle('instagram_eng_hel.pkl')
