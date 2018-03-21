
#!pip install textblob

from textblob import TextBlob
import pandas as pd
import numpy as np

# Import csv to DataFrame
df = pd.read_csv('yelp.csv')

# Remove nulls
df = df[df.text.notnull()]

# Check head
#df.head()

# Check length
len(df)

# Functions to assess sentiment and get accuracy
def get_sentiment(text):
    '''Return polarity and subjectivity of given input text.'''
    text = TextBlob(text)
    return(text.sentiment)

def get_label(sentiment):
    '''Assign label based on polarity and subjectivity score'''
    if sentiment[0] <= 0.1:
        return('Negative')
    if sentiment[0] > 0.1 and sentiment[1] >= 0.4:
        return('Positive')
    else:
        return("Can't tell")

# The above decision boundaries were the result of manual tweaking, but run gridsearch to get optimal boundaries

def run_blob(dataframe,text_column,label_column):
    '''Get sentiment, get label, and get accuracy of labels'''
    dataframe['textblob'] = [get_sentiment(x) for x in dataframe[text_column]]
    dataframe['predict'] = [get_label(x) for x in dataframe['textblob']]
    accuracy = len(dataframe[dataframe['predict'] == dataframe[label_column]])/len(dataframe)
    return(accuracy)

### Run on Subset
# Let's start with a small sample
df_small = df.sample(500)

# Check head
# df_small.head()

# Apply to subset and return accuracy
run_blob(df_small,'text','sentiment')

# Check head with newly created columns
# df_small.head()

# Check the original categories and counts of the small dataframe
df_small.predict.value_counts()

# Check the prediction categories and counts of the small dataframe
df_small.sentiment.value_counts()

### Run on Full Dataset
# Let's apply it to the full set and get the accuracy
run_blob(df,'text','sentiment')

# Check the original categories and counts of the full dataframe
df.sentiment.value_counts()

# Check the prediction categories and counts of the full dataframe
df.predict.value_counts()