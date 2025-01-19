#!/usr/bin/env python
# coding: utf-8

# # Amenities: Features Importance

# ## Amenities: Data Cleaning 

# In[171]:


# Importing required libraries
import pandas as pd
import numpy as np
from numpy.random import seed
seed(123)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


# ### Import the data

# In[172]:


raw_df = pd.read_csv('listings.csv')
print(f"The dataset contains {len(raw_df)} Airbnb listings")
pd.set_option('display.max_columns', len(raw_df.columns)) # To view all columns
pd.set_option('display.max_rows', 100)
raw_df.head(3)


# ### Drop the initial values columns 

# In[173]:


cols_to_drop = ['listing_url', 'scrape_id', 'last_scraped', 'name','description', 'neighborhood_overview','host_id', 'host_url', 'host_name', 'host_location', 'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_verifications', 'calendar_last_scraped']
df = raw_df.drop(cols_to_drop, axis=1)


# ### Checking the Null Values 

# In[174]:


df.isna().sum()


# ### Dropping irrelevant columns, due to NaN Values 

# In[175]:


df.drop(['host_acceptance_rate', 'neighbourhood_group_cleansed', 'license'], axis=1, inplace=True)

df.set_index('id', inplace=True) # The id will be used as the index, as t


# In[176]:


print(sum((df.host_listings_count == df.host_total_listings_count) == False))
df.loc[((df.host_listings_count == df.host_total_listings_count) == False)][:5]

host_listings_count and host_total_listings_count are the same in all but 59 cases. These cases are those where the value is NaN. Therefore one of these columns can be dropped. Other columns which split these into type of property will also be dropped, as they will be highly correlated (one will be the total of the others).
# ### Continue to drop columns not needed 

# In[177]:


df.drop(['host_total_listings_count', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms'], axis=1, inplace=True)


# In[178]:


df.drop([ 'latitude', 'longitude', 'neighbourhood'], axis=1, inplace=True)


# In[179]:


df.drop(['minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'], axis=1, inplace=True)


# In[180]:


# Number of rows without a value for multiple host-related columns
len(df[df.loc[ :,[ 'host_is_superhost', 'host_listings_count', 'host_identity_verified'] ].isnull().sum(axis=1) == 5])


# ### Cleaning Amenities column  

# In[181]:


# Example of amenities listed
df.amenities[:1].values


# In[182]:


# Creating a set of all possible amenities
amenities_list = list(df.amenities)
amenities_list_string = " ".join(amenities_list)
amenities_list_string = amenities_list_string.replace('[', '')
amenities_list_string = amenities_list_string.replace(']', ',')
amenities_list_string = amenities_list_string.replace('"', '')
amenities_set = [x.strip() for x in amenities_list_string.split(',')]
amenities_set = set(amenities_set)
amenities_set


# In[183]:


### Locking the amenities to simplify them 


# In[184]:


df.loc[df['amenities'].str.contains('24-hour check-in'), 'check_in_24h'] = 1
df.loc[df['amenities'].str.contains('Air conditioning|Central air conditioning'), 'air_conditioning'] = 1
df.loc[df['amenities'].str.contains('Amazon Echo|Apple TV|Game console|Netflix|Projector and screen|Smart TV'), 'high_end_electronics'] = 1
df.loc[df['amenities'].str.contains('BBQ grill|Fire pit|Propane barbeque'), 'bbq'] = 1
df.loc[df['amenities'].str.contains('Balcony|Patio'), 'balcony'] = 1
df.loc[df['amenities'].str.contains('Beach view|Beachfront|Lake access|Mountain view|Ski-in/Ski-out|Waterfront'), 'nature_and_views'] = 1
df.loc[df['amenities'].str.contains('Bed linens'), 'bed_linen'] = 1
df.loc[df['amenities'].str.contains('Breakfast'), 'breakfast'] = 1
df.loc[df['amenities'].str.contains('TV'), 'tv'] = 1
df.loc[df['amenities'].str.contains('Coffee maker|Espresso machine'), 'coffee_machine'] = 1
df.loc[df['amenities'].str.contains('Cooking basics'), 'cooking_basics'] = 1
df.loc[df['amenities'].str.contains('Dishwasher|Dryer|Washer'), 'white_goods'] = 1
df.loc[df['amenities'].str.contains('Elevator'), 'elevator'] = 1
df.loc[df['amenities'].str.contains('Exercise equipment|Gym|gym'), 'gym'] = 1
df.loc[df['amenities'].str.contains('Family/kid friendly|Children|children'), 'child_friendly'] = 1
df.loc[df['amenities'].str.contains('parking'), 'parking'] = 1
df.loc[df['amenities'].str.contains('Garden|Outdoor|Sun loungers|Terrace'), 'outdoor_space'] = 1
df.loc[df['amenities'].str.contains('Host greets you'), 'host_greeting'] = 1
df.loc[df['amenities'].str.contains('Hot tub|Jetted tub|hot tub|Sauna|Pool|pool'), 'hot_tub_sauna_or_pool'] = 1
df.loc[df['amenities'].str.contains('Internet|Pocket wifi|Wifi'), 'internet'] = 1
df.loc[df['amenities'].str.contains('Long term stays allowed'), 'long_term_stays'] = 1
df.loc[df['amenities'].str.contains('Pets|pet|Cat(s)|Dog(s)'), 'pets_allowed'] = 1
df.loc[df['amenities'].str.contains('Private entrance'), 'private_entrance'] = 1
df.loc[df['amenities'].str.contains('Safe|Security system'), 'secure'] = 1
df.loc[df['amenities'].str.contains('Self check-in'), 'self_check_in'] = 1
df.loc[df['amenities'].str.contains('Smoking allowed'), 'smoking_allowed'] = 1
df.loc[df['amenities'].str.contains('Step-free access|Wheelchair|Accessible'), 'accessible'] = 1
df.loc[df['amenities'].str.contains('Suitable for events'), 'event_suitable'] = 1


# ### Replacing the nulls with 0 and then remove amenities columns that are less then 10% of the listings 

# In[185]:


# Replacing nulls with zeros for new columns
cols_to_replace_nulls = df.iloc[:,41:].columns
df[cols_to_replace_nulls] = df[cols_to_replace_nulls].fillna(0)

# Produces a list of amenity features where one category (true or false) contains fewer than 10% of listings
infrequent_amenities = []
for col in df.iloc[:,41:].columns:
    if df[col].sum() < len(df)/10:
        infrequent_amenities.append(col)
print(infrequent_amenities)

# Dropping infrequent amenity features
df.drop(infrequent_amenities, axis=1, inplace=True)

# Dropping the original amenity feature
df.drop('amenities', axis=1, inplace=True)


# In[186]:


df.columns[40:] ##showing the remaining listings 


# In[187]:


### Changing the price into float and remove comma 


# In[188]:


df.price = df.price.str[1:-3]
df.price = df.price.str.replace(",", "")
df.price = df.price.astype('int64')


# In[189]:


df.head()


# In[190]:


df.info() #showing the finalized data


# ### Made a new copy of the cleaned data with the only the price and chosen amenities in each column

# In[191]:


df2 = df[["price", "balcony", "bed_linen","breakfast", "tv", "coffee_machine", "cooking_basics", "white_goods","elevator", "parking", "host_greeting", "internet", "long_term_stays", "private_entrance"]]


# In[192]:


df2 #the amenities are encoded to nominal variables 


# In[193]:


def clean_dataset(df3):
    assert isinstance(df3, pd.DataFrame), "df needs to be a pd.DataFrame"  
    df.dropna(inplace=True)
    indices_to_keep = ~df3.isin([np.nan, np.inf, -np.inf]).any(1)
    return df2[indices_to_keep].astype(np.float64) #to ensure that there's no null values and the feature importance result can work


# ## Feature Importance Using Decision Tree

# In[194]:


# Putting feature variable to X
X = df3.drop('price',axis=1)

# Putting response variable to y
y = df3['price']


# ### Import Train-Test split 

# In[195]:


from sklearn.model_selection import train_test_split


# In[196]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


# ### Import Decision Tree Classifier 

# In[197]:


from sklearn.tree import DecisionTreeClassifier


# In[198]:


dt = DecisionTreeClassifier(max_depth=3,random_state=43)
dt.fit(X_train, y_train)


# In[199]:


### Import confusion matrix and accuract score 


# In[200]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[201]:


dt = DecisionTreeClassifier(random_state=43)
dt.fit(X_train,y_train)


# In[202]:


y_train_pred = dt.predict(X_train) #predict the X_train 
y_test_pred = dt.predict(X_test)  #predict the X_test 


# In[203]:


print(accuracy_score(y_train, y_train_pred)) #the accuracy score of train model 
confusion_matrix(y_train, y_train_pred) #the confusion matrix of train model


# In[204]:


print(accuracy_score(y_test, y_test_pred)) #the accuracy score of test model 
confusion_matrix(y_test, y_test_pred) #the confusion matrix of test model 


# In[205]:


# Let's check the overall accuracy.
trainaccuracy= accuracy_score(y_train, y_train_pred)
testaccuracy= accuracy_score(y_test, y_test_pred)

confusion_TRN = confusion_matrix(y_train, y_train_pred)
confusion_TST = confusion_matrix(y_test, y_test_pred)


# ### Checking the Confusion Matrix(true positve/negative, false positive/negative) 

# In[206]:


TP = confusion_TRN[1,1] # true positive 
TN = confusion_TRN[0,0] # true negatives
FP = confusion_TRN[0,1] # false positives
FN = confusion_TRN[1,0] # false negatives

TP_TST = confusion_TST[1,1] # true positive 
TN_TST = confusion_TST[0,0] # true negatives
FP_TST = confusion_TST[0,1] # false positives
FN_TST = confusion_TST[1,0] # false negatives

trainsensitivity= TP / float(TP+FN)
trainspecificity= TN / float(TN+FP)

testsensitivity= TP_TST / float(TP_TST+FN_TST)
testspecificity= TN_TST / float(TN_TST+FP_TST)

# Let us compare the values obtained for Train & Test:
print('-'*30)
print('On Train Data')
print('-'*30)
print("Accuracy    : {} %".format(round((trainaccuracy*100),2)))
print("Sensitivity : {} %".format(round((trainsensitivity*100),2)))
print("Specificity : {} %".format(round((trainspecificity*100),2)))
print('-'*30)
print('On Test Data')
print('-'*30)
print("Accuracy    : {} %".format(round((testaccuracy*100),2)))
print("Sensitivity : {} %".format(round((testsensitivity*100),2)))
print("Specificity : {} %".format(round((testspecificity*100),2)))
print('-'*30)


# ### Import the GridSearchCV to check the best parameters for Decision Tree 

# In[207]:


from sklearn.model_selection import GridSearchCV


# In[208]:


dt_hp = DecisionTreeClassifier(random_state=43)

params = {'max_depth':[3,5,7,10],
          'min_samples_leaf':[5,10,15,20],
          'min_samples_split':[10,12,18,20],
          'criterion':['gini','entropy']}
GS = GridSearchCV(estimator=dt_hp,param_grid=params,cv=5,n_jobs=-1, verbose=True, scoring='accuracy')


# In[209]:


GS.fit(X_train, y_train)


# In[210]:


print('Best Parameters:',GS.best_params_,end='\n\n')
print('Best Score:',GS.best_score_)


# In[211]:


dt_hp = DecisionTreeClassifier(max_depth= 9,min_samples_leaf= 25, min_samples_split=5 ,random_state=43)


# In[212]:


dt_hp.fit(X_train, y_train)


# In[213]:


y_train_pred = dt_hp.predict(X_train)
y_test_pred = dt_hp.predict(X_test)

# Let's check the overall accuracy.
trainaccuracy= accuracy_score(y_train, y_train_pred)
testaccuracy= accuracy_score(y_test, y_test_pred)

confusion_TRN = confusion_matrix(y_train, y_train_pred)
confusion_TST = confusion_matrix(y_test, y_test_pred)

TP = confusion_TRN[1,1] # true positive 
TN = confusion_TRN[0,0] # true negatives
FP = confusion_TRN[0,1] # false positives
FN = confusion_TRN[1,0] # false negatives

TP_TST = confusion_TST[1,1] # true positive 
TN_TST = confusion_TST[0,0] # true negatives
FP_TST = confusion_TST[0,1] # false positives
FN_TST = confusion_TST[1,0] # false negatives

trainsensitivity= TP / float(TP+FN)
trainspecificity= TN / float(TN+FP)

testsensitivity= TP_TST / float(TP_TST+FN_TST)
testspecificity= TN_TST / float(TN_TST+FP_TST)

# Let us compare the values obtained for Train & Test:
print('-'*30)
print('On Train Data')
print('-'*30)
print("Accuracy    : {} %".format(round((trainaccuracy*100),2)))
print("Sensitivity : {} %".format(round((trainsensitivity*100),2)))
print("Specificity : {} %".format(round((trainspecificity*100),2)))
print('-'*30)
print('On Test Data')
print('-'*30)
print("Accuracy    : {} %".format(round((testaccuracy*100),2)))
print("Sensitivity : {} %".format(round((testsensitivity*100),2)))
print("Specificity : {} %".format(round((testspecificity*100),2)))
print('-'*30)


# ### Creating a dictionary of amenity features and show their importance values

# In[214]:


# let's create a dictionary of features and their importance values
feat_dict= {}
for col, val in sorted(zip(X_train.columns, dt_hp.feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val


# In[215]:


feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})


# In[216]:


feat_df


# ### Plotting the Amenity Features and Its Importance using Matplotlib 

# In[217]:


values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x < max(values)) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Amenities features to predict Prices(Using Decision Tree)')
plt.show()


# ## Feature importance using Random Forest Regression 

# In[218]:


# Putting feature variable to X
X = df3.drop('price',axis=1)

# Putting response variable to y
y = df3['price']


# ### Importing Train-test split 

# In[219]:


from sklearn.model_selection import train_test_split


# In[220]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


# ### Importing RandomForestClassifier 

# In[224]:


from sklearn.ensemble import RandomForestRegressor


# In[225]:


# Creating and fitting the model
RF = RandomForestRegressor(n_estimators=2000, max_depth=40,min_samples_split = 50,
                           max_leaf_nodes = 70,max_features = 5)

RF.fit(X_train, y_train)
# Predicting the training and testing sets
trainPredictin_RF = RF.predict(X_train)
testPredictin_RF =RF.predict(X_test)


# In[226]:


# The model accurracies and validation analysis
trainPredictin_RF = RF.predict(X_train)
testPredictin_RF =RF.predict(X_test)

print('Random Forest MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, trainPredictin_RF),
        mean_squared_error(y_test, testPredictin_RF)))
print('Random Forest R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, trainPredictin_RF),
        r2_score(y_test, testPredictin_RF)))


# ### Creating feature importance of the amenities 

# In[227]:


importancesRF = RF.feature_importances_
feat_imp1 = pd.DataFrame(importancesRF, columns=['Importance'], index= X_train.columns)
feat_imp1.sort_values(by=['Importance'], ascending= False, inplace=True)
feat_imp1


# In[228]:


feat_imp = pd.DataFrame({'Importance':RF.feature_importances_})  #creating a data of the features and importance 
feat_imp['Feature'] = X_train.columns

feat_imp.sort_values(by='Importance', inplace=True)
feat_imp.sort_values(by='Importance', ascending=False, inplace=True)

feat_imp = feat_imp[["Feature", "Importance"]]
feat_imp


# ### Plotting the Amenity Features and Its Importance using Matplotlib 

# In[229]:


values = feat_imp.Importance    
idx = feat_imp.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x < max(values)) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title= 'Important Amenities features to predict Prices(Using Random Forest Regression)')
plt.show()

