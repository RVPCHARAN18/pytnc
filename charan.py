#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup
from newspaper import Article  
import csv 
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize 
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
url = "https://timesofindia.indiatimes.com/world"
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html5lib')
table = soup.findAll('a', attrs = {'class':'w_img'})
news=[]
for row in table: 
    if not row['href'].startswith('http'):
        news.append('https://timesofindia.indiatimes.com'+row['href'])

import nltk
nltk.download('punkt')
df=[]
for i in news:
    article = Article(i, language="en")
    article.download() 
    article.parse() 
    article.nlp() 
    data={}
    data['Title']=article.title
    data['Text']=article.text
    data['Summary']=article.summary
    data['Keywords']=article.keywords
    df.append(data)

dataset=pd.DataFrame(df)
dataset.head()
FILEPATH=r"C:\Users\pavan\Downloads\crawl.csv"

def TrainTestSplit(X, Y, R=0, test_size=0.2):
    return train_test_split(X, Y, test_size=test_size, random_state=R)
def clean_cols(data):
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(index=str, columns=clean_col_map)
full_data = clean_cols(pd.read_csv(FILEPATH))
train_set, test_set = train_test_split(full_data, test_size=0.20, random_state=42)

x_train = train_set.drop(['url','shares','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','abs_title_subjectivity','abs_title_sentiment_polarity'], axis=1)
y_train = train_set['shares']

x_test = test_set.drop(['url','shares', 'self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','abs_title_subjectivity','abs_title_sentiment_polarity'], axis=1)
y_test = test_set['shares']

clf = RandomForestRegressor(random_state=42)
clf.fit(x_train, y_train)


from nltk.tokenize import word_tokenize 
def tokenize(text):
    text=text
    return word_tokenize(text)

pos_words=[]
neg_words=[]
def rates(words):
    words=words
    pos=words[0]
    neg=words[1]
    all_words=words
    global_rate_positive_words=(len(pos)/len(all_words))/100
    global_rate_negative_words=(len(neg)/len(all_words))/100
    pol_pos=[]
    pol_neg=[]
    for i in pos:
        a=TextBlob(i)
        pol_pos.append(a.sentiment.polarity)
        avg_positive_polarity=a.sentiment.polarity
    for j in neg:
        a1=TextBlob(j)
        pol_neg.append(a1.sentiment.polarity)
        avg_negative_polarity=a1.sentiment.polarity

def polar(words):
    tokens=token(words)
    for i in tokens:
        analysis=TextBlob(i)
        p=analysis.sentiment.polarity
        if p>0:
            pos_words.append(i)
        if p<0:
            neg_words.append(i)
    return pos_words,neg_words

df2=[]
for i in news:
    pred_info={}
    article = Article(i, language="en")
    article.download() 
    article.parse()
    a2=TextBlob(article.text)
    polarity=a2.sentiment.polarity
    title_a=TextBlob(article.title)
    pred_info['text']=article.text
    pred_info['n_tokens_title']=len(tokenize(article.title))
    pred_info['n_tokens_content']=len(tokenize(article.text))
    pred_info['num_hrefs']=article.html.count("https://timesofindia.indiatimes.com")
    pred_info['num_imgs']=len(article.images)
    pred_info['title_subjectivity']=title_a.sentiment.subjectivity
    pred_info['title_sentiment_polarity']=title_a.sentiment.polarity
    df2.append(pred_info)
pred_df=pd.DataFrame(df2)
pred_test=pred_df.drop(['text'],axis=1)
pred_df.head()
test2=pd.DataFrame(clf.predict(pred_test),pred_df['text'])
test2.reset_index(level=0, inplace=True)
test2 = test2.rename(index=str, columns={"index": "News", 0: "Virality"})
print(test2)

