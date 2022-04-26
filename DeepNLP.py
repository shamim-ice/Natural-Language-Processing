import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.utils.validation import check_memory
stop = set(stopwords.words('english'))
from wordcloud import WordCloud, STOPWORDS

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#print(check_memory(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('/content/Natural-Language-Processing/Sheet_2.csv', encoding='latin-1')
print(data.head())