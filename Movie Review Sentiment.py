#!/usr/bin/env python
# coding: utf-8

# # Import requried libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words()
# Import necessary libraries

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv("IMDB Dataset.csv")


# In[3]:


df


# # EDA

# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df['sentiment'].value_counts()


# # LabelEncoding

# In[9]:


df


# In[10]:


def cleaning(review):        
    # converting to lowercase, removing URL links, special characters, punctuations...
    review = review.lower() # converting to lowercase
    
    review = re.sub('https?://\S+|www\.\S+', '', review) # removing URL links
    review = re.sub(r"\b/d+\b", "", review) # removing number 
    review = re.sub('<.*?>+', '', review) # removing special characters, 
    review = re.sub('\n', '', review)
    review = re.sub('[’“”…]', '', review)
   
    #removing emoji: 
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    review = emoji_pattern.sub(r'',review)   

   # removing short form: 
    
    review=re.sub("isn't",'is not',review)
    review=re.sub("he's",'he is',review)
    review=re.sub("wasn't",'was not',review)
    review=re.sub("there's",'there is',review)
    review=re.sub("couldn't",'could not',review)
    review=re.sub("won't",'will not',review)
    review=re.sub("they're",'they are',review)
    review=re.sub("she's",'she is',review)
    review=re.sub("There's",'there is',review)
    review=re.sub("wouldn't",'would not',review)
    review=re.sub("haven't",'have not',review)
    review=re.sub("That's",'That is',review)
    review=re.sub("you've",'you have',review)
    review=re.sub("He's",'He is',review)
    review=re.sub("what's",'what is',review)
    review=re.sub("weren't",'were not',review)
    review=re.sub("we're",'we are',review)
    review=re.sub("hasn't",'has not',review)
    review=re.sub("you'd",'you would',review)
    review=re.sub("shouldn't",'should not',review)
    review=re.sub("let's",'let us',review)
    review=re.sub("they've",'they have',review)
    review=re.sub("You'll",'You will',review)
    review=re.sub("i'm",'i am',review)
    reviewt=re.sub("we've",'we have',review)
    review=re.sub("it's",'it is',review)
    review=re.sub("don't",'do not',review)
    review=re.sub("that´s",'that is',review)
    review=re.sub("I´m",'I am',review)
    review=re.sub("it’s",'it is',review)
    review=re.sub("she´s",'she is',review)
    review=re.sub("he’s'",'he is',review)
    review=re.sub('I’m','I am',review)
    review=re.sub('I’d','I did',review)
    review=re.sub("he’s'",'he is',review)
    review=re.sub('there’s','there is',review)
    
     
    return review

    
dt = df['review'].apply(cleaning)


# In[11]:


pd.set_option("max.colwidth",0)


# In[12]:


#dt


# In[13]:


df['sentiment']


# In[14]:


dt = pd.DataFrame(dt)  
dt['sentiment']=df['sentiment']
dt


# In[15]:


dt['no_sw'] = dt['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[16]:


X=df['review']
y=df['sentiment']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


label = LabelEncoder()
df['sentiment'] = label.fit_transform(df['sentiment'])


# ###  Define the pipeline for Naive Bayes Multinomial

# In[19]:


nb_pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])


# In[20]:


# Train the Naive Bayes Multinomial model
nb_pipeline.fit(X_train, y_train)


# In[21]:


# Predictions on the test set
nb_predictions = nb_pipeline.predict(X_test)


# In[22]:


# Evaluate Naive Bayes Multinomial model
print("Naive Bayes Multinomial Results:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Classification Report:\n", classification_report(y_test, nb_predictions))


# #### Naive Bayes : accuray score  0.86 

# ### Define the pipeline for Logistic Regression

# In[23]:


lr_pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])


# In[24]:


# Train the Logistic Regression model
lr_pipeline.fit(X_train, y_train)


# In[25]:


# Predictions on the test set
lr_predictions = lr_pipeline.predict(X_test)


# In[26]:


# Evaluate Logistic Regression model
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Classification Report:\n", classification_report(y_test, lr_predictions))


# #### Logistic regression : accuracy score 0.88

# ### Define the pipeline for XGBoost

# In[27]:


xgb_pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('clf', GradientBoostingClassifier())
])


# In[28]:


# Train the XGBoost model
xgb_pipeline.fit(X_train, y_train)


# In[29]:


# Predictions on the test set
xgb_predictions = xgb_pipeline.predict(X_test)


# In[30]:


# Evaluate XGBoost model
print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_test, xgb_predictions))
print("Classification Report:\n", classification_report(y_test, xgb_predictions))


# #### XGBoost : accuracy score 0.81

# ## Support Vector Classifier

# In[31]:


from sklearn.svm import SVC


# In[32]:


svm_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  
    ('svm', SVC())  
])


# In[33]:


svm_pipeline.fit(X_train, y_train)


# In[34]:


svm_pred = svm_pipeline.predict(X_test)


# In[35]:


print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))


# #### Support Vector Classifier : accuracy score 0.88

# In[36]:


##Logistic regression and svc 


# In[ ]:




