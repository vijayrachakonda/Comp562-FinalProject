import numpy as numpy
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import *

df = pd.read_csv('../Downloads/wine-reviews/winemag-data-130k-v2.csv')

# Selecting only top ten win varieties to be classified
top_ten = df['variety'].value_counts()[:10].index.tolist()
df = df[df['variety'].isin(top_ten)]

# Cleaning up dataframe to only show the relevant columns (i.e. description and variety)
cols_to_keep = ['description', 'variety']
df = df.loc[:, cols_to_keep]

df_x = df['description']
df_y = df['variety']


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%", "flavor", "wine"]
stops = text.ENGLISH_STOP_WORDS.union(punc)
count_vect = CountVectorizer(stop_words=stops, tokenizer=tokenize)
x_train_counts = count_vect.fit_transform(x_train)


# making vector of counts for each word in each row in dataframe
# tfidf_vec = TfidfVectorizer(stop_words="english")
# tfidf_vec_train = tfidf_vec.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
# print(tfidf_vec_train.shape)

# tfidf_vec = TfidfVectorizer(vocabulary=tfidf_vec.vocabulary_, stop_words="english")
# tfidf_vec_test = tfidf_vec.fit_transform(x_test)
# print(tfidf_vec_test.shape)

model = MultinomialNB().fit(x_train_tfidf, y_train)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)

predicted_vareities = model.predict(x_test_tfidf)

print("Accuracy: ", accuracy_score(y_test, predicted_vareities))