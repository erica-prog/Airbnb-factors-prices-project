#!/usr/bin/env python
# coding: utf-8

# ## Reviews Part I: Sentiment Analysis 

# In[267]:


import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# In[268]:


cols = ["comments"]
reviews= pd.read_csv("reviews.csv", usecols = cols)
reviews["comments"] = reviews["comments"].astype(str) #spliting the text into words
reviews.drop(reviews.index[10001:1043004], inplace=True)
reviews.head()


# ### Preprocessing/Cleaning of the Data 

# In[269]:


#clean the text
#create a function to clean the comments

def CleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\S]+', '', text)
    
    return text

#cheaning the text
reviews['comments'] = reviews['comments'].apply(CleanTxt)

#shows clean text
reviews.head()


# In[270]:


new_reviews = reviews.copy()


# ### Calculating the Subjectivty and Polarity using TextBlob

# In[271]:


#CREATE A FUNCTION TO GET THE SUBJECTIVITY 
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#create a function to get the polarity 

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#create new columns
new_reviews['Subjectivity']= new_reviews['comments'].apply(getSubjectivity)
new_reviews['Polarity']= new_reviews['comments'].apply(getPolarity)

#shows new 
new_reviews.head(10)


# ### Computing the Negative, Neutral and Positve Analysis

# In[272]:


from tqdm.notebook import tqdm


# In[273]:


#compute a function to compute the negative, neutral and postive analysis
with tqdm(total=new_reviews.shape[0]) as pbar:
    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
        pbar.update(1)


new_reviews['Analysis']= new_reviews['Polarity'].apply(getAnalysis)


#show dataframe
new_reviews


# In[274]:


new_reviews['Analysis'].value_counts()


# In[275]:


new_reviews.loc[[3138, 4000], :]


# Shows that most of the 10000 selected comments are postive, compared to negative and neutal comments

# ### Plotting the subjectivity and polarity scatterplot

# In[276]:


plt.figure(figsize =(8,6))
for i in range (0, new_reviews.shape[0]):
    plt.scatter(new_reviews['Polarity'][i], new_reviews['Subjectivity'][i], color = 'Green')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


# In[277]:


cols = [
    "amenities",
    "price",
    "review_scores_rating",
    "reviews_per_month"
]

listing = pd.read_csv("listings.csv", usecols=cols)


# In[278]:


airbnb_reviews = pd.read_csv("reviews.csv")


# In[279]:


new_airbnb_reviews = pd.concat([airbnb_reviews, listing], axis='columns') 


# In[280]:


new_airbnb_reviews.head()


# In[281]:


new_airbnb_reviews.shape


# In[282]:


new_airbnb_reviews.dtypes


# In[293]:


new_airbnb_reviews["price"].min()


# ## Reviews Part II: Comparing Prices to Reviews

# ### Changing price into floats and removing $

# In[283]:


# Remove $ from price before conversion to float
new_airbnb_reviews['price'] = new_airbnb_reviews['price'].str.replace("$", " ")
# Print header to make sure change was done
new_airbnb_reviews['price'].head()


# In[284]:


new_airbnb_reviews.price = new_airbnb_reviews.price.replace('[\$,]', '', regex=True).astype(float)


# ### Plotting the Review score rating and Price using Matplotlib

# In[285]:


import matplotlib.pyplot as plt

plt.scatter(new_airbnb_reviews['price'],new_airbnb_reviews['review_scores_rating'])
plt.xlabel('price $')
plt.ylabel('review score')
plt.title('Scatter plot of review rating vs. price')
plt.show()


# ## Reviews III: Price vs Reviews_per_month Scatter Graph

# In[286]:


import seaborn as sns # ploting the data


# In[287]:


sns.set_palette("muted")
x = 'reviews_per_month'
y = 'price'

title = 'Price relation to number of review per month for Properties under £175'
data_filtered = new_airbnb_reviews.loc[(new_airbnb_reviews['price'] < 175) & (new_airbnb_reviews['reviews_per_month'] < 30)]
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=data_filtered)
plt.title(title)
plt.ioff()

title = 'Price relation to number of review per month for Properties more than £175'
data_filtered = new_airbnb_reviews.loc[new_airbnb_reviews['price'] > 175]
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=data_filtered)
plt.title(title)
plt.ioff()


# ## Amenities: The Top 20 Amenities 

# ### Preprocessing the Amenities column & Plotting of the amenities using Matplotlib

# In[288]:


new_airbnb_reviews = new_airbnb_reviews[new_airbnb_reviews['amenities'].notnull()] #AttributeError: 'float' object has no attribute 'split'


# In[289]:


# Create an array of all unique amenities across all rows by splitting it
# measure the top 20 amenities
pd.Series(np.concatenate(new_airbnb_reviews['amenities'].map(lambda amns: amns.split(","))))    .value_counts().head(20)    .plot(kind='bar')
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
plt.show()

