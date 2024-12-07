# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:43:24 2024

@author: sinth
"""

import pandas as pd
import nltk
df=pd.read_csv("spam.csv",encoding="latin-1")
df.head(5)
df.shape
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'class','v2':'sms'},inplace=True)
df.sample(5)
df.groupby('class').describe()
df=df.drop_duplicates(keep='first')
df.groupby('class').describe()
df["Length"]=df["sms"].apply(len)
df.head(5)
df.hist(column='Length',by='class',bins=50)
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('punkt')
ps=PorterStemmer()
df.head(5)
import string
def clean_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)
df['sms_cleaned']=df['sms'].apply(clean_text)
df.head(5)
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vec=TfidfVectorizer(max_features=3000)
x=tf_vec.fit_transform(df['sms_cleaned']).toarray()
x.shape
y=df['class'].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
y_pred=model.predict(X_test)
print(accuracy_score(Y_test,y_pred))