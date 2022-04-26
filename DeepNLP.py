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

from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()

#print(check_memory(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('/content/Natural-Language-Processing/Sheet_2.csv', encoding='latin-1')
#print(data.head())


def cloud(text):
  wordcloud  = WordCloud(background_color='red', stopwords = stop).generate(' '.join([i for i in text.str.upper()]))
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.title('Resume Response')

cloud(data['resume_text'])


Encode = preprocessing.LabelEncoder()

data['label'] = Encode.fit_transform(data['class'])
#print(data.head)

x = data['resume_text']
y = data['label']
vect = CountVectorizer()
x = vect.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,
random_state=25)



NB.fit(x_train, y_train)
y_predict = NB.predict(x_test)
acc = metrics.accuracy_score(y_test, y_predict)
print(acc)



