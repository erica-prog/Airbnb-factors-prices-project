#!/usr/bin/env python
# coding: utf-8

# ## Reviews Part IV: NLTK Top Words in All Comments 

# In[21]:


import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud


# In[22]:


cols = ["comments"]
reviews= pd.read_csv("reviews.csv", usecols = cols)
reviews["comments"] = reviews["comments"].astype(str) #put the comments into strings 
reviews.head()


# In[23]:


reviews.comments.head() 


# In[24]:


reviews.comments.values[1]


# ### Import Natural Language Processing Libraries 

# In[25]:


#Natural Language processing 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud


# ### Preprocessing of the reviews data 

# In[27]:


#Preprocessing of the data  
reviews = reviews[reviews['comments'].notnull()]
                                        #Take out empty comments
reviews['comments'] = reviews['comments'].str.replace('\d+','')
                                        #remove numbers 
reviews['comments'] = reviews['comments'].str.lower()
                                        #lowercase
reviews['comments'] = reviews['comments'].str.replace('\r\n',"")
                                        #remove windows new line
stop_english=stopwords.words("english")
reviews['comments'] = reviews['comments'].apply(lambda x: " ".join([i for i in x.split() 
                                                      if i not in (stop_english)]))
                                        #remove all the stop words with nltk library
reviews['comments'] = reviews['comments'].str.replace('[^\w\s]'," ")
                                        #remove all punctuation
reviews['comments'] = reviews['comments'].str.replace('\s+', ' ')
                                        #replace x spaces by one space 


    
reviews['comments'].values[1]
                                        #print the comment index1 one more time 


# ### Top 10 common words in the comments 

# In[30]:


#Top 10 common words in the comments with CountVectorizer()
texts= reviews.comments.tolist()

vec = CountVectorizer().fit(texts)
bag_of_words = vec.transform(texts)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

cvec_df = pd.DataFrame.from_records(words_freq, columns= ['words', 'counts']).sort_values(by="counts", ascending=False)
cvec_df.head(10)


# In[29]:


#Create the word cloud from the file we have 
cvec_dict = dict(zip(cvec_df.words, cvec_df.counts))

wordcloud = WordCloud(width=800, height=400)
wordcloud.generate_from_frequencies(frequencies=cvec_dict)
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

