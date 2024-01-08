#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# nltk is one of the most useful libraries when it comes to nlp
get_ipython().system('pip install nltk')
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Preprocessing and evaluation
from nltk.corpus import stopwords
from tqdm import tqdm


# In[ ]:





# In[3]:


df=pd.read_excel("hotel_reviews.xlsx")
df.head()


# In[4]:


df.values


# In[5]:


df.tail()


# In[6]:


df.shape


# # Cleaning the data

# In[7]:


df.info()


# In[8]:


df['Rating'].value_counts()


# As you can see from above details, there is no null values in this dataset

# In[9]:


# look  describe data set
df["Rating"].describe().round(2)


# In[10]:


#The average a round 4 , it is good


# In[11]:


# any duplicate data
df.duplicated().sum()


# # DATA VISUALIZATION

# In[12]:


sns.countplot(data=df, x='Rating', palette='flare').set_title('Rating Distribution Across Dataset')


# The percentage rating
# 
# 5 = 44%
# 
# 4 = 30%
# 
# 3 = 10%
# 
# 2 = 9%
# 
# 1 = 7%
# 

# * Rating 5 = 44% -> satisfy
# 
# We see the people satisfy = rating 5 in this words
# 
# In general the hotel - room - night - beach - restaurant and food and drink - bed - pool - locations.
# 
# * Rating 4 = 30% -> satisfy
# 
# We see the people satisfy = rating 4 in this words the same rating 5 but plus ...
# 
# beautiful hotel - staff friendly - service - street.
# 
# * Rating 2 = 9% -> unsatisfied
# 
# We see the people unsatisfied = rating 2 in this words the same rating 5 & 4 but plus ...
# 
# Hotel - staff - beach - srevice - disk - stay - shower
# 
# * Rating 1 = 7% -> unsatisfied
# 
# We see the people unsatisfied = rating 1 in this words the same rating 5 & 4 but plus ...
# 
# room - hotel - place - staff - door - check in - sleep - toilet - resort -water.

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.catplot(x='Rating', kind="count", aspect = 2, data=df, order = df['Rating'].value_counts().index)


# In[14]:


label_data= df['Rating'].value_counts()
labels_name = ['5','4','3','2','1']
explode=(0.05,0.05,0.05,0.05,0.05)
plt.figure(figsize=(12,7))
patches, texts, pcts= plt.pie(label_data, labels= labels_name, colors=['blue','red','yellow','cyan','Green'], pctdistance=0.65,shadow=True,
                             startangle=90, explode=explode, autopct='%1.1f%%',
                             textprops={'fontsize':12,'color':'black','weight':'bold','family':'serif'})
plt.setp(pcts,color='black')
hfont={'weight':'bold','family':'serif'}
plt.title('HOTEL RATING DISTRIBUTION',size=30,**hfont)

centre_circle=plt.Circle((0,0),0.40,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# In[15]:


# Visualizing results (Barchart for top 10 nouns + verbs)
df[0:10].plot.bar(x='Review',figsize=(12,8),title='Top 10 nouns and verbs');


# In[16]:


df.plot()


# In[17]:


df.hist()


# In[18]:


sns.heatmap(df.isna().sum().to_frame(),annot=True,cmap='mako')
plt.xlabel('Amount Missing',fontsize=15)
plt.show()


# In[19]:


df.boxplot()


# # Text Mining

# In[20]:


#Calculating the count of words and length of Review in each review
df["word_count"]= df.Review.apply(lambda x: len(str(x).split(' ')))
df["Review_length"] = df.Review.apply(len)


# In[21]:


df.describe().T


# In[22]:


#Reviewing the outliers:
df1= df.sort_values(by = 'Review_length',ascending=False)
df1.head(20)


# In[ ]:





# In[23]:


#shape of Dataset:
print(f'there are {df1.shape[0]} reviews and {df1.shape[1]} features in the dataset')


# # Number of Words

# In[24]:


#Number of Words in single tweet
df['word_count'] = df['Review'].apply(lambda x: len(str(x).split(" ")))
df[['Review','word_count']].head(10)


# # Number of Characters

# In[25]:


#Number of characters in single tweet
df['char_count'] = df['Review'].str.len() ## this also includes spaces
df[['Review','char_count']].head(10)


# # Average Word Length

# In[26]:


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

df['avg_word'] = df['Review'].apply(lambda x: avg_word(x))
df[['Review','avg_word']].head(5)


# # # Number of stopwords

# In[27]:


stop = stopwords.words('english')

df['stopwords'] = df['Review'].apply(lambda x: len([x for x in x.split() if x in stop]))
df[['Review','stopwords']].head(5)


# # Number of Special Characters

# In[28]:


df['hastags'] = df['Review'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
df[['Review','hastags']].head()


# # Number of Numerics

# In[29]:


df['numerics'] = df['Review'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['Review','numerics']].head()


# # Number of Upper Case Words

# In[30]:


df['upper'] = df['Review'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['Review','upper']].head()


# # Number of Upper Case Words

# In[31]:


df['upper'] = df['Review'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['Review','upper']].head()


# # Pre - Processing

# # Lower Case

# In[32]:


df['Review'] = df['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Review'].head()


# # Common word removal

# In[33]:


freq = pd.Series(' '.join(df['Review']).split()).value_counts()[:10]
freq


# In[34]:


freq = list(freq.index)
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['Review'].head()


# # Rare Words Removal

# In[35]:


freq = pd.Series(' '.join(df['Review']).split()).value_counts()[-10:]
freq


# In[36]:


freq = list(freq.index)
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['Review'].head()


# # Spelling correction

# In[37]:


pip install -U textblob


# In[38]:


from textblob import TextBlob


# In[39]:


df['Review'][:5].apply(lambda x: str(TextBlob(x).correct()))


# # Wordlist

# In[40]:


TextBlob(df['Review'][1]).words


# In[41]:


TextBlob(df['Review'][2]).words


# In[42]:


TextBlob(df['Review'][3]).words


# In[43]:


nltk.download('punkt')


# # Stemming

# In[44]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
df['Review'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# # Using VADER

# In[45]:


pip install vaderSentiment


# In[46]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[47]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Initialize the Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

# Assuming df is your DataFrame

# Define a function to perform sentiment analysis using VADER
def get_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)

    if sentiment_scores['compound'] > 0:
        return 'positive'
    elif sentiment_scores['compound'] == 0:
        return 'neutral'
    else:
        return 'negative'

# Group the DataFrame by 'Rating_name'
grouped_df = df.groupby('Rating')

for Rating_name, group in grouped_df:
    Rating_sentiments = group['Review'].apply(get_sentiment)
    sentiment_counts = Rating_sentiments.value_counts()

    # Do something with sentiment_counts, for example, print the results
    print(f"Sentiment analysis for {Rating_name}:")
    print(sentiment_counts)
    print("\n")


# In[48]:


df.head(10)


# In[49]:


reviews_df = df.sample(frac=0.1, replace=False, random_state=42)


# In[50]:


# remove 'No Negative' or 'No Positive' from text
reviews_df["Review"] = reviews_df["Review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))


# In[51]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the vader_lexicon resource
nltk.download('vader_lexicon')

# Assuming your DataFrame is named reviews_df
sid = SentimentIntensityAnalyzer()
reviews_df["sentiments"] = reviews_df["Review"].apply(lambda x: sid.polarity_scores(x))
reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)


# In[52]:


# add number of characters column
reviews_df["nb_chars"] = reviews_df["Review"].apply(lambda x: len(x))

# add number of words column
reviews_df["nb_words"] = reviews_df["Review"].apply(lambda x: len(x.split(" ")))


# In[53]:


reviews_df.head()


# In[54]:


print(reviews_df.shape)


# # Creating WordCloud for Positive and Negative reviews

# In[55]:


df['Sentiment'] = np.where(df['Rating'] == 3, 'Neutral', np.where(df['Rating'] > 3, 'Positive', 'Negative'))


# In[56]:


df.head(3)


# In[57]:


Positive_sentiment = df['Sentiment'] == 'Positive'
Positive_df = df[Positive_sentiment]


# In[58]:


# Creating a word cloud visualization of the Positive sentiment category
cloud = WordCloud(width=800, height=600).generate(" ".join(df[Positive_sentiment]['Review']))
plt.figure(figsize=(16, 10))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[ ]:


#Creating Negative sentiment category
Negative_sentiment = df['Sentiment'] == 'Negative'
Negative_df = df[Negative_sentiment]


# In[ ]:


#Creating a word cloud visualization of the Negative sentiment category
cloud=WordCloud(width=800, height=600).generate(" ".join(df[Negative_sentiment]['Review']))
plt.figure(figsize=(16,10))
plt.imshow(cloud)
plt.axis('off')


# In[ ]:



sns.countplot(x=df['Sentiment'])
plt.show()


# # standardize_text

# In[ ]:


# function for cleaning Review
def standardize_text(df, field):
    df[field] = df[field].str.replace(r"http\S+", "")
    df[field] = df[field].str.replace(r"http","")
    df[field] = df[field].str.replace(r"@/S+","")
    df[field] = df[field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[field] = df[field].str.replace(r"@"," at ")
    df[field] = df[field].str.lower()
    return df


# In[ ]:


standardize_text(df,"Review")


# # clean_text

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_text(text):
    stop = set(stopwords.words('english'))
    punc = set(punctuation)
    bad_tokens = stop.union(punc)
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    word_tokens = [t for t in tokens if t.isalpha()]
    clean_tokens = [lemma.lemmatize(t.lower()) for t in word_tokens if t.lower() not in bad_tokens]
    return " ".join(clean_tokens)


# In[ ]:


# Applying text preprocessing methods to df['Review']
df['Review'] = df['Review'].apply(clean_text)


# In[ ]:


df.head(5)


# In[ ]:


df.values


# # 1. Creating WordClouds

# In[ ]:


from wordcloud import WordCloud
wc = WordCloud(width=800,
               height=500,
               background_color='black',
               min_font_size=10)
wc.generate(''.join(df['Review']))
plt.figure(figsize=(10,10))
plt.imshow(wc)
plt.axis('off')
plt.show()


# # Split into test and train

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Review, df.Sentiment)


# In[ ]:





# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
v= CountVectorizer()
x_train_vec = v.fit_transform(x_train)
x_test_vec = v.transform(x_test)


# # Use a Classification model

# # Support Vector Machines (SVM)

# In[ ]:


from sklearn import svm
from  pickle import dump
from pickle import load
import pickle

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(x_train_vec, y_train)


# In[ ]:


with open('svm_model.pkl', 'wb') as model_file:
    dump(clf_svm, model_file)

# Load the model from the file
with open('svm_model.pkl', 'rb') as model_file:
    loaded_svm_model = load(model_file)


# # Teat accuracy

# In[ ]:


clf_svm.score(x_test_vec, y_test)


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, clf_svm.predict(x_test_vec), average= None)


# In[ ]:


rev=['This place was beautiful, cant wait to come back']
rev_vec =v.transform(rev)
clf_svm.predict(rev_vec)


# In[ ]:


rev=['ok just looks nice modern outside']
rev_vec =v.transform(rev)
clf_svm.predict(rev_vec)


# # RandomForestClassifier

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Text preprocessing and feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Model building
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(x_train_tfidf, y_train)

# Prediction
y_pred = random_forest_classifier.predict(x_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")


# In[ ]:


from sklearn.metrics import f1_score
y_pred = random_forest_classifier.predict(x_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
# Classification Report
report = classification_report(y_test, y_pred)


print(f"F1 Score: {f1}")
print("Classification Report:\n", report)


# In[ ]:


# New review
new_review = ['ok nothing special charge diamond member']

# Transform the new review using the same TfidfVectorizer
new_review_tfidf = tfidf_vectorizer.transform(new_review)

# Prediction
y_pred = random_forest_classifier.predict(new_review_tfidf)

# Print the predicted sentiment
print("Predicted Sentiment:", y_pred[0])


# # Multinomial Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

# Model building - Multinomial Naive Bayes
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(x_train_tfidf, y_train)

# Prediction
y_pred_nb = naive_bayes_classifier.predict(x_test_tfidf)

# Evaluation
accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')
report_nb = classification_report(y_test, y_pred_nb)

# Print results for Multinomial Naive Bayes
print("Multinomial Naive Bayes:")
print(f"Accuracy: {accuracy_nb}")
print(f"F1 Score: {f1_nb}")
print("Classification Report:\n", report_nb)


# # plotting the accuracy and f1 score

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


models = ['Random Forest', 'SVM', 'Multinomial Naive Bayes']
accuracy_values = [accuracy, clf_svm.score(x_test_vec, y_test), accuracy_nb]
f1_values = [f1, f1_score(y_test, clf_svm.predict(x_test_vec), average='weighted'), f1_nb]

# Plotting Accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(models, accuracy_values, color=['blue', 'orange', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')

# Plotting F1 Score
plt.subplot(1, 2, 2)
plt.bar(models, f1_values, color=['blue', 'orange', 'green'])
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison')

plt.tight_layout()
plt.show()


# By seeing the plot we can say the Support Vector Machines (SVM) is giving good accuracy

# In[ ]:




